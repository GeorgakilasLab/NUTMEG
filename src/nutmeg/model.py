import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear

class NUTMEG(nn.Module):


    def __init__(self,
                 hidden_dim: int,
                 num_classes: int,
                 num_genes: int,
                 *,
                 mode: str = "multiome",
                 heads: int = 1,
                 dropout: float = 0.0,
                 per_gene_hidden: int = None,       
                 include_presence_bit: bool = True,
                 return_intermediates: bool = False):        
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        assert mode in ["multiome", "rna_only"]
        
        self.H = hidden_dim
        self.G = num_genes
        self.C = num_classes
        self.mode = mode
       

        if per_gene_hidden is None:
            per_gene_hidden = max(8, hidden_dim // 2)

        self.return_intermediates = return_intermediates
        self.include_presence_bit = include_presence_bit

        per_head = hidden_dim // heads
        self.act  = nn.ELU()
        self.drop = nn.Dropout(dropout) if (dropout is not None and dropout > 0) else nn.Identity()

        self.proj_gene_in = Linear(-1, hidden_dim, bias=False)

        if mode == "multiome":
            self.proj_cre_in  = Linear(-1, hidden_dim, bias=False)

            # Two GAT hops
            self.gene_to_cre = GATConv((-1, -1), per_head, heads=heads, concat=True,
                                   add_self_loops=False, dropout=dropout)
            self.cre_to_gene = GATConv((-1, -1), per_head, heads=heads, concat=True,
                                   add_self_loops=False, dropout=dropout)

            self.norm_cre  = nn.LayerNorm(hidden_dim)

        else:
            self.gene_to_gene = GATConv((-1, -1), hidden_dim, heads=1, concat=False,
                                        add_self_loops=False, dropout=dropout)
        
        self.norm_gene = nn.LayerNorm(hidden_dim)

        # -------- Per-gene MLP projector --------
        in_dim = hidden_dim + (1 if include_presence_bit else 0)
        self.per_gene_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, per_gene_hidden, bias=True),
            nn.ELU(),
            nn.Dropout(dropout) if (dropout is not None and dropout > 0) else nn.Identity(),
            nn.Linear(per_gene_hidden, 1, bias=True),
        )

        # -------- Classifier over genes: --------
        self.classifier = Linear(self.G, self.C, bias=True)


    def _dense_scatter_embeddings(self, x: torch.Tensor, b: torch.Tensor, gidx: torch.Tensor, B: int):
        N, H = x.size(0), x.size(1)
        out_flat = x.new_zeros(B * self.G * H)                  
        base = b * (self.G * H) + gidx * H                      
        ch   = torch.arange(H, device=x.device).view(1, H)      
        idx  = base.unsqueeze(1) + ch                           
        out_flat.index_add_(0, idx.reshape(-1), x.reshape(-1))
        return out_flat.view(B, self.G, H)                      
    
    def forward(self, data):
        
        tf_feats   = data['gene'].tf_feats.float()
        gene_feats = data['gene'].gene_feats.float()
        
        gene_batch = getattr(data['gene'], 'batch', None)
        if gene_batch is None or gene_batch.numel() == 0:
            gene_batch = torch.zeros(gene_x.size(0), dtype=torch.long, device=gene_x.device)
        B = int(gene_batch.max().item()) + 1 if gene_batch.numel() else 1

        if self.mode == "multiome":
            cre_in     = data['cre'].x.float()
            ei_g2c = data['gene','binds','cre'].edge_index
            ei_c2g = data['cre','regulates','gene'].edge_index
            
            cre_msg, (ei_g2c_out, alpha_g2c) = self.gene_to_cre(
                (tf_feats, cre_in), ei_g2c, return_attention_weights=True
            )
            cre_x = self.drop(self.act(self.norm_cre(cre_msg + self.proj_cre_in(cre_in))))

            # --- Hop 2: CRE -> Gene ---
            gene_msg, (ei_c2g_out, alpha_c2g) = self.cre_to_gene(
                (cre_x, gene_feats), ei_c2g, return_attention_weights=True
            )

            attn = {
            "g2c": {"edge_index": ei_g2c_out, "alpha": alpha_g2c},
            "c2g": {"edge_index": ei_c2g_out, "alpha": alpha_c2g},
            }
            
        else:
            ei_g2g = data['gene','regulates','gene'].edge_index

            gene_msg, (ei_g2g_out, alpha_g2g) = self.gene_to_gene(
                (tf_feats, gene_feats), ei_g2g, return_attention_weights=True
            )

            attn = {
                "t2g": {"edge_index": ei_g2g_out, "alpha": alpha_g2g},
            }
        
        gene_x = self.drop(self.act(self.norm_gene(gene_msg + self.proj_gene_in(gene_feats))))  


        # Keep only targets 
        is_target = getattr(data['gene'], 'is_target', None)
        if is_target is not None and is_target.view(-1).bool().any():
            mask_nodes = is_target.view(-1).bool()
        else:
            mask_nodes = torch.ones(gene_x.size(0), dtype=torch.bool, device=gene_x.device)

        # Dense align to [B,G,H]
        x    = gene_x[mask_nodes]
        b    = gene_batch[mask_nodes]
        gidx = data['gene'].kept_idx[mask_nodes].long()               
        assert gidx.min() >= 0 and gidx.max() < self.G, "kept_idx out of [0..G-1]"
        aligned = self._dense_scatter_embeddings(x, b, gidx, B)       

        # Presence mask (1 if gene exists in this cell, else 0)
        presence = (aligned.abs().sum(dim=-1) > 0).float()            

        # ---- Per-gene MLP to scalar scores ----
        aligned_in = torch.cat([aligned, presence.unsqueeze(-1)], dim=-1) \
                     if self.include_presence_bit else aligned          
        scores = self.per_gene_mlp(aligned_in).squeeze(-1)              
        scores = scores * presence                                      

        # ---- classifier across genes ----
        logits = self.classifier(scores)                                 

        if self.return_intermediates:
            pack = {
                "aligned":    aligned,     
                "presence":   presence,    
                "aligned_in": aligned_in,  
            }
            return logits, scores, attn, pack


        return logits, scores, attn
