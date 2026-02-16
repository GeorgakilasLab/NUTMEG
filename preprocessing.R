suppressPackageStartupMessages({
  library(R6)

  library(Signac)
  library(Seurat)

  library(EnsDb.Hsapiens.v86)
  library(BSgenome.Hsapiens.UCSC.hg38)

  library(JASPAR2020)
  library(TFBSTools)
  library(motifmatchr)

  library(ggplot2)
  library(patchwork)

  library(GenomicRanges)
  library(ensembldb)
  library(Matrix)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(reticulate)
})

GRNPrep <- R6Class(
  "GRNPrep",
  lock_objects = FALSE,
  public = list(
    

    initialize = function(
      mode = c("RNA_ONLY", "MULTIOME"),
      source = c("10X_H5", "CUSTOM", "SEURAT_OBJECT"),
      experiment = "exp_01",
      tenx_h5_path     = NULL,
      tenx_frag_path   = NULL,
      custom_rna_path  = NULL,
      custom_atac_path = NULL,
      custom_frag_path  = NULL,
      seed = 42  
    ) {
      self$mode   <- mode
      self$source <- match.arg(source)
      self$experiment <- experiment
      self$base_dir   <- experiment

      self$tenx_h5_path     <- tenx_h5_path
      self$tenx_frag_path   <- tenx_frag_path
      self$custom_rna_path  <- custom_rna_path
      self$custom_atac_path <- custom_atac_path
      self$custom_frag_path <- custom_frag_path
      self$obj              <- NULL  

      self$seed <- seed
      set.seed(self$seed)

      self$data_dir        <- file.path(self$base_dir, "data")
      self$interm_dir      <- file.path(self$base_dir, "interm")
      self$csv_dir         <- file.path(self$interm_dir, "csv")
      self$plot_dir        <- file.path(self$interm_dir, "plots")
      self$feature_mtx_dir <- file.path(self$interm_dir, "feature_mtx")
      self$rds_dir         <- file.path(self$base_dir, "rds")

      dir.create(self$csv_dir,         showWarnings = FALSE, recursive = TRUE)
      dir.create(self$plot_dir,        showWarnings = FALSE, recursive = TRUE)
      dir.create(self$feature_mtx_dir, showWarnings = FALSE, recursive = TRUE)
      dir.create(self$rds_dir,         showWarnings = FALSE, recursive = TRUE)

      self$plots   <- list()
      self$reports <- list()

      private$validate_inputs()

      message(
        "\n[GRNPrep] Initialized (experiment: ", self$base_dir, ")\n",
        "  - mode   = ", self$mode, "\n",
        "  - source = ", self$source, "\n",
        "  - seed   = ", self$seed, "\n",
        "\n[GRNPrep] Output layout created under", normalizePath(self$base_dir),"\n",
        "  - CSV:         ", normalizePath(self$csv_dir), "\n",
        "  - plots:       ", normalizePath(self$plot_dir), "\n",
        "  - feature_mtx: ", normalizePath(self$feature_mtx_dir), "\n",
        "  - rds:         ", normalizePath(self$rds_dir), "\n"
      )

      invisible(self)
    },

    # ==========================================================
    # 0) Inject a Seurat object
    # ==========================================================
    set_object = function(seurat_obj, mode = c("AUTO", "RNA_ONLY", "MULTIOME"), annotation = NULL) {
      mode <- match.arg(mode)
      if (!inherits(seurat_obj, "Seurat")) stop("set_object(): seurat_obj must be a Seurat object.")

      inferred_mode <- if ("ATAC" %in% Assays(seurat_obj)) "MULTIOME" else "RNA_ONLY"
      if (mode == "AUTO") mode <- inferred_mode

      if (!("RNA" %in% Assays(seurat_obj))) stop("set_object(): object must contain RNA assay.")
      if (mode == "MULTIOME" && !("ATAC" %in% Assays(seurat_obj))) {
        stop("set_object(): mode='MULTIOME' requires ATAC assay in the object.")
      }

      self$obj    <- seurat_obj
      self$mode   <- mode
      self$source <- "SEURAT_OBJECT"

      if (!is.null(annotation)) self$annotation <- annotation
      if (self$mode == "MULTIOME" && is.null(self$annotation)) {
        self$annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
        seqlevelsStyle(self$annotation) <- "UCSC"
      }

      message(
        "\n[set_object] Injected Seurat object\n",
        "  - mode: ", self$mode, "\n",
        "  - assays: ", paste(Assays(self$obj), collapse = ", "), "\n",
        "  - cells: ", ncol(self$obj), "\n"
      )
      invisible(self)
    },

    # ==========================================================
    # 1) LOAD
    # ==========================================================
    load = function(keep_standard_chromosomes = TRUE) {
      set.seed(self$seed)

      if (self$source == "SEURAT_OBJECT" && !is.null(self$obj)) {
        message("[load] Using injected Seurat object; skipping load().")
        return(invisible(self))
      }

      message("\n[load] Loading data... (source=", self$source, ", mode=", self$mode, ")")

      self$annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
      seqlevelsStyle(self$annotation) <- "UCSC"

      if (self$source == "10X_H5") {
        if (is.null(self$tenx_h5_path)) stop("[load] tenx_h5_path is NULL.")
        counts <- Read10X_h5(self$tenx_h5_path)
        rna_counts <- counts$`Gene Expression`

        if (self$mode == "MULTIOME") {
          peak_mat <- counts$Peaks
          fragpath <- self$tenx_frag_path
        } else {
          peak_mat <- NULL
          fragpath <- NULL
        }

      } else if (self$source == "CUSTOM") {
        if (is.null(self$custom_rna_path) || !file.exists(self$custom_rna_path)) {
          stop("[load] custom_rna_path is NULL or not found.")
        }
        rna_counts <- readRDS(self$custom_rna_path)

        if (self$mode == "MULTIOME") {
          if (is.null(self$custom_atac_path) || !file.exists(self$custom_atac_path)) {
            stop("[load] custom_atac_path is NULL or not found.")
          }
          peak_mat <- readRDS(self$custom_atac_path)
          fragpath <- self$custom_frag_path
        } else {
          peak_mat <- NULL
          fragpath <- NULL
        }
      } else {
        stop("[load] source='SEURAT_OBJECT' requires set_object().")
      }

      obj <- CreateSeuratObject(counts = rna_counts, assay = "RNA")

      if (self$mode == "MULTIOME") {
        if (is.null(fragpath)) stop("[load] MULTIOME requires fragments path.")
        if (keep_standard_chromosomes) {
          std.chr <- paste0("chr", c(1:22, "X", "Y"))
          keep_peaks <- grepl(paste0("^(", paste(std.chr, collapse = "|"), "):"), rownames(peak_mat))
          peak_mat <- peak_mat[keep_peaks, , drop = FALSE]
        }

        obj[["ATAC"]] <- CreateChromatinAssay(
          counts     = peak_mat,
          sep        = c(":", "-"),
          fragments  = fragpath,
          annotation = self$annotation
        )
      }

      self$obj <- obj
      message("[load] Done. Cells=", ncol(self$obj))
      invisible(self)
    },

    # ==========================================================
    # 2) QC 
    # ==========================================================
    qc = function(save_plots = TRUE, filename = "qc_hist_before_filtering.png", force = FALSE) {
      if (is.null(self$obj)) self$load()
      obj <- self$obj
      md <- obj@meta.data

      if (!force && all(c("log_nCount_RNA", "percent.mt") %in% colnames(md))) {
        message("[qc] QC metrics already present; skipping qc().")
        return(invisible(self$plots$qc_before_filtering))
      }

      message("\n[qc] Computing QC metrics... (mode=", self$mode, ")")

      DefaultAssay(obj) <- "RNA"
      obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
      obj$log_nCount_RNA  <- log10(obj$nCount_RNA)

      hist_list <- list(
        ggplot(obj@meta.data, aes(x = log_nCount_RNA)) +
          geom_histogram(bins = 50, fill = "steelblue", color = "black") +
          theme_minimal(base_size = 14) +
          ggtitle("log10(nCount_RNA)") +
          theme(axis.title.y = element_blank(), axis.title.x = element_blank()),
        ggplot(obj@meta.data, aes(x = percent.mt)) +
          geom_histogram(bins = 50, fill = "steelblue", color = "black") +
          theme_minimal(base_size = 14) +
          ggtitle("percent.mt") +
          theme(axis.title.y = element_blank(), axis.title.x = element_blank())
      )

      if (self$mode == "MULTIOME") {
        DefaultAssay(obj) <- "ATAC"
        obj <- NucleosomeSignal(obj)
        obj <- TSSEnrichment(obj)

        obj$log_nCount_ATAC       <- log10(obj$nCount_ATAC)
        obj$log_TSS.enrichment    <- log10(obj$TSS.enrichment)
        obj$log_nucleosome_signal <- log10(obj$nucleosome_signal)

        hist_list <- c(hist_list, list(
          ggplot(obj@meta.data, aes(x = log_nCount_ATAC)) +
            geom_histogram(bins = 50, fill = "steelblue", color = "black") +
            theme_minimal(base_size = 14) +
            ggtitle("log10(nCount_ATAC)") +
            theme(axis.title.y = element_blank(), axis.title.x = element_blank()),
          ggplot(obj@meta.data, aes(x = log_TSS.enrichment)) +
            geom_histogram(bins = 50, fill = "steelblue", color = "black") +
            theme_minimal(base_size = 14) +
            ggtitle("log10(TSS.enrichment)") +
            theme(axis.title.y = element_blank(), axis.title.x = element_blank()),
          ggplot(obj@meta.data, aes(x = log_nucleosome_signal)) +
            geom_histogram(bins = 50, fill = "steelblue", color = "black") +
            theme_minimal(base_size = 14) +
            ggtitle("log10(Nucleosome signal)") +
            theme(axis.title.y = element_blank(), axis.title.x = element_blank())
        ))
      }

      qc_plot <- wrap_plots(hist_list, ncol = 3) +
        plot_annotation(
          title = if (self$mode == "MULTIOME") "QC metrics (Multiome) before filtering"
          else "QC metrics (RNA-only) before filtering",
          theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))
        )

      self$obj <- obj
      self$plots$qc_before_filtering <- qc_plot
      if (save_plots) private$save_plot_if(qc_plot, filename, 14, 10, 300)

      message("[qc] Done. Plot saved -> ", file.path(self$plot_dir, filename))
      invisible(qc_plot)
    },

    # ==========================================================
    # 3) FILTER
    # ==========================================================
    filter = function(
      rna_log_ncount_min = 3,
      rna_log_ncount_max = 4,
      rna_mt_pct_max     = 20,
      atac_log_ncount_min = 3,
      atac_log_ncount_max = 4.5,
      atac_log_nucleo_max = 0.3,
      atac_log_tss_min    = 0.0,
      min_cells_per_gene = 3,
      min_cells_per_peak = 10
    ) {
      if (is.null(self$obj)) stop("[filter] Run $load() then $qc() before filtering.")
      obj <- self$obj

      message("\n[filter] Filtering cells & features... (mode=", self$mode, ")")

      md <- obj@meta.data
      need_rna <- c("log_nCount_RNA", "percent.mt")
      if (length(setdiff(need_rna, colnames(md))) > 0) stop("[filter] Missing RNA QC metrics. Run $qc() first.")

      if (self$mode == "MULTIOME") {
        need_atac <- c("log_nCount_ATAC", "log_TSS.enrichment", "log_nucleosome_signal")
        if (length(setdiff(need_atac, colnames(md))) > 0) stop("[filter] Missing ATAC QC metrics. Run $qc() first.")
      }

      n_cells_before <- ncol(obj)

      keep <- if (self$mode == "RNA_ONLY") {
        with(md, log_nCount_RNA > rna_log_ncount_min & log_nCount_RNA < rna_log_ncount_max & percent.mt < rna_mt_pct_max)
      } else {
        with(md,
             log_nCount_RNA > rna_log_ncount_min & log_nCount_RNA < rna_log_ncount_max & percent.mt < rna_mt_pct_max &
               log_nCount_ATAC > atac_log_ncount_min & log_nCount_ATAC < atac_log_ncount_max &
               log_nucleosome_signal < atac_log_nucleo_max & log_TSS.enrichment > atac_log_tss_min
        )
      }

      keep <- !is.na(keep) & is.finite(keep) & keep
      obj  <- obj[, keep]

      rna_counts <- private$get_layer(obj, assay = "RNA", layer = "counts")
      gene_counts   <- Matrix::rowSums(rna_counts > 0)
      genes_to_keep <- names(gene_counts[gene_counts > min_cells_per_gene])
      obj[["RNA"]]  <- subset(obj[["RNA"]], features = genes_to_keep)

      if (self$mode == "MULTIOME") {
        atac_counts <- private$get_layer(obj, assay = "ATAC", layer = "counts")
        peak_counts   <- Matrix::rowSums(atac_counts > 0)
        peaks_to_keep <- names(peak_counts[peak_counts > min_cells_per_peak])
        obj[["ATAC"]] <- subset(obj[["ATAC"]], features = peaks_to_keep)
      }

      self$obj <- obj
      message("[filter] Done. Cells: ", n_cells_before, " -> ", ncol(obj))
      invisible(self)
    },

    # ==========================================================
    # 5) PREPROCESS
    # ==========================================================
    preprocess = function(
      n_sct_features = 2000,
      n_pca_dims     = 50,
      n_lsi_dims     = 20,
      resolution     = 0.5,
      save_plots     = TRUE,

      # Labeling parameters
      label_mode          = c("clusters", "labels"),
      labels_column_name  = "cell_label",
      labels_file         = NULL,
      labels_sep          = ",",

      final_rds_filename = "pbmc.rds"
    ) {
      if (is.null(self$obj)) stop("[preprocess] Run $load()/$qc()/$filter() first.")
      obj <- self$obj
      label_mode <- match.arg(label_mode)
      set.seed(self$seed)

      message("\n[preprocess] Preprocessing + labeling (mode=", self$mode, ", labels=", label_mode, ")")

      if (!("SCT" %in% Assays(obj))) {
        DefaultAssay(obj) <- "RNA"
        message("[preprocess] Running SCTransform (n_features=", n_sct_features, ") ...")
        set.seed(self$seed)
        obj <- SCTransform(obj, variable.features.n = n_sct_features, seed.use = self$seed, verbose = FALSE)
        message("[preprocess] Running PCA ...")
        set.seed(self$seed)
        obj <- RunPCA(obj, seed.use = self$seed, verbose = FALSE)
      } else if (!("pca" %in% names(obj@reductions))) {
        message("[preprocess] Running PCA ...")
        set.seed(self$seed)
        obj <- RunPCA(obj, seed.use = self$seed, verbose = FALSE)
      } else {
        message("[preprocess] SCT+PCA already present; skipping.")
      }

      if (self$mode == "RNA_ONLY") {
        obj <- FindNeighbors(obj, reduction = "pca", dims = 1:n_pca_dims, verbose = FALSE)
        set.seed(self$seed)
        obj <- RunUMAP(obj, reduction = "pca",dims = 1:n_pca_dims, metric = "correlation", seed.use = self$seed, verbose = FALSE)
        obj <- FindClusters(obj, resolution = resolution, verbose = FALSE)

        p <- DimPlot(obj, group.by = "seurat_clusters", label = TRUE, repel = TRUE) + NoLegend()
        self$plots$umap_clusters <- p
        if (save_plots) private$save_plot_if(p, "umap_rna_only_clusters.png", 8, 6, 300)

      } else {
        if (!("ATAC" %in% Assays(obj))) stop("[preprocess] MULTIOME requires ATAC assay.")

        DefaultAssay(obj) <- "ATAC"
        if (!("lsi" %in% names(obj@reductions))) {
          obj <- FindTopFeatures(obj, min.cutoff = "q75")
          obj <- RunTFIDF(obj)
          obj <- RunSVD(obj, n = n_lsi_dims)
        } else {
          message("[preprocess] LSI already present; skipping TFIDF/SVD.")
        }

        obj <- FindMultiModalNeighbors(
          object = obj,
          reduction.list = list("pca", "lsi"),
          dims.list      = list(1:n_pca_dims, 2:n_lsi_dims),
          modality.weight.name = "RNA.weight",
          verbose = FALSE
        )

        set.seed(self$seed)
        obj <- RunUMAP(obj, nn.name = "weighted.nn", assay = "RNA", metric = "correlation", seed.use = self$seed, verbose = FALSE)
        obj <- FindClusters(obj, graph.name = "wsnn", algorithm = 4, resolution = resolution, verbose = FALSE)

        p <- DimPlot(obj, group.by = "seurat_clusters", label = TRUE, repel = TRUE, reduction = "umap") + NoLegend()
        self$plots$umap_clusters <- p
        if (save_plots) private$save_plot_if(p, "umap_wnn_clusters.png", 8, 6, 300)
      }

      message("[preprocess] Adding labels (mode=", label_mode, ")")
      if (label_mode == "clusters") {
        if (!"seurat_clusters" %in% colnames(obj@meta.data)) stop("[preprocess] No clusters found.")
        obj[[labels_column_name]] <- as.character(obj$seurat_clusters)
      } else {  # "labels" mode
        if (is.null(labels_file)) stop("[preprocess] labels mode needs labels_file.")
        if (!file.exists(labels_file)) stop("[preprocess] labels_file not found: ", labels_file)
    
        lab_df <- read.csv(labels_file, sep = labels_sep, stringsAsFactors = FALSE, check.names = FALSE)
        if (ncol(lab_df) < 2) stop("[preprocess] labels_file needs >=2 cols: barcode,label.")
        colnames(lab_df)[1:2] <- c("barcode", "label")
    
        common_bcs <- intersect(colnames(obj), lab_df$barcode)
        if (length(common_bcs) == 0) stop("[preprocess] No barcode overlap.")
    
        lab_df <- lab_df[match(common_bcs, lab_df$barcode), , drop = FALSE]
        obj <- obj[, common_bcs]
        obj[[labels_column_name]] <- lab_df$label
      }
  
      Idents(obj) <- labels_column_name

      labeled_cells <- which(!is.na(obj[[labels_column_name]]))
      if (length(labeled_cells) == 0) stop("[preprocess] No labeled cells found!")
  
      message("[preprocess] Subsetting: ", length(labeled_cells), "/", ncol(self$obj), " labeled cells")
      obj <- obj[, labeled_cells]
      labels_df <- data.frame(
        barcode    = Cells(obj),                              # or colnames(obj)
        cell_label = as.character(obj[[labels_column_name]][, 1]),
        stringsAsFactors = FALSE
      )

      labels_csv_path <- file.path(self$csv_dir, "labels.csv")
      write.csv(labels_df, labels_csv_path, row.names = FALSE, quote = FALSE)
      message("[preprocess] labels.csv saved: ", normalizePath(labels_csv_path))
      
      final_path <- file.path(self$rds_dir, final_rds_filename)
      saveRDS(obj, final_path)
      message("[preprocess] Labeled object saved: ", normalizePath(final_path))


      self$obj <- obj
      invisible(self)
    },

    # ==========================================================
    # 5) CREATE GRN EXPORTS 
    # ==========================================================
    grn_exports = function(
      motif_raw_threshold = 5,
      jaspar_species = "Homo sapiens",
      jaspar_all_versions = FALSE,

      promoter_window_bp = 2000,

      genes_use = NULL,  # default = HVGs from SCT
      peaks_use = NULL,  # default = HV peaks from ATAC VariableFeatures
      link_peak_assay = "ATAC",
      link_expression_assay = "SCT",

      peaks_genes_filename    = "peaks_genes.csv",
      tf_peak_scores_filename = "tf_peak_scores.csv"
    ) {
      if (is.null(self$obj)) {
        stop("[grn_exports] Run load/qc/filter/preprocess first (or inject a preprocessed object).")
      }
      obj <- self$obj
      set.seed(self$seed)

      message("\n[grn_exports] Building base GRN exports... (mode=", self$mode, ")")

      if (!("SCT" %in% names(obj@assays))) {
        stop("[grn_exports] SCT assay not found. Run preprocess() (SCTransform) first.")
      }

      hvg <- if (is.null(genes_use)) VariableFeatures(obj, assay = "SCT") else genes_use
      message("[grn_exports] Gene-side set = ", length(hvg), " genes (HVGs unless genes_use provided).")

      # ----------------------------------------------------------
      # Step A: define CRE->gene edges 
      # ----------------------------------------------------------
      if (self$mode == "MULTIOME") {

        if (!("ATAC" %in% names(obj@assays))) stop("[grn_exports] MULTIOME requires ATAC assay.")

        hvp <- if (is.null(peaks_use)) VariableFeatures(obj, assay = "ATAC") else peaks_use
        message("[grn_exports] Peak-side set = ", length(hvp), " peaks (HV peaks unless peaks_use provided).")

        DefaultAssay(obj) <- "ATAC"
        main.chroms <- standardChromosomes(BSgenome.Hsapiens.UCSC.hg38)
        keep.peaks  <- which(as.character(seqnames(granges(obj))) %in% main.chroms)
        obj[["ATAC"]] <- subset(obj[["ATAC"]], features = rownames(obj[["ATAC"]])[keep.peaks])
        obj <- RegionStats(obj, genome = BSgenome.Hsapiens.UCSC.hg38)

        message("[grn_exports] Running LinkPeaks (", link_peak_assay, " -> ", link_expression_assay, ") ...")
        obj <- LinkPeaks(
          object = obj,
          peak.assay = link_peak_assay,
          expression.assay = link_expression_assay,
          genes.use = hvg
        )

        links_df <- Links(obj) |> as.data.frame()
        links_df <- links_df %>% dplyr::filter(peak %in% hvp)

        if (nrow(links_df) == 0) {
          stop("[grn_exports] No peak-gene links after filtering to selected peaks (peaks_use/HV peaks).")
        }
        message("[grn_exports] LinkPeaks links after peak filtering: ", nrow(links_df))

        # Compute nearest TSS distance
        links_df <- links_df %>%
          mutate(peak_norm = gsub(":", "-", peak)) %>%
          separate(
            peak_norm, into = c("peak_chr","peak_start","peak_end"),
            sep = "-", convert = TRUE, remove = FALSE
          ) %>%
          mutate(peak_center = (peak_start + peak_end) / 2)

        tx_all <- transcripts(EnsDb.Hsapiens.v86, filter = GeneNameFilter(unique(links_df$gene)))
        tx_df  <- as.data.frame(tx_all) %>%
          mutate(
            seqnames = paste0("chr", as.character(seqnames)),
            seqnames = ifelse(seqnames == "chrMT", "chrM", seqnames),
            tss      = ifelse(strand == "+", start, end)
          ) %>%
          filter(seqnames %in% c(paste0("chr", 1:22), "chrX", "chrY")) %>%
          select(tx_id, gene_name, seqnames, strand, tss)

        links_with_tss <- links_df %>%
          inner_join(
            tx_df,
            by = c("gene" = "gene_name", "peak_chr" = "seqnames"),
            relationship = "many-to-many"
          ) %>%
          mutate(distance_bp = abs(peak_center - tss))

        nearest_tss <- links_with_tss %>%
          group_by(peak, gene) %>%
          slice_min(order_by = distance_bp, n = 1, with_ties = FALSE) %>%
          ungroup() %>%
          mutate(
            peak_type       = ifelse(distance_bp <= promoter_window_bp, "promoter", "distal"),
            distance_scaled = as.numeric(scale(log1p(distance_bp)))
          ) %>%
          select(peak, gene, distance_bp, distance_scaled, peak_type, score, zscore, pvalue)

        links_out <- nearest_tss
        cre_gr <- private$peaks_to_granges(unique(links_out$peak))

        message("[grn_exports] MULTIOME CRE->Gene edges: ", nrow(links_out),
                " | unique CREs=", length(unique(links_out$peak)),
                " | unique genes=", length(unique(links_out$gene)))

      } else {

        # ----------------------------------------------------------
        # RNA_ONLY: PROMOTER-ONLY pseudo-CREs 
        # ----------------------------------------------------------
        message("[grn_exports] RNA_ONLY: using PROMOTER-ONLY pseudo-CREs.")

        prom_df <- private$make_promoter_cre(
          obj = obj,
          promoter_window_bp = promoter_window_bp
        )
        if (!all(c("chr","start","end","gene","peak") %in% colnames(prom_df))) {
          stop("[grn_exports] Internal error: promoter CRE table missing required columns.")
        }

        cre_gr <- GRanges(
          seqnames = prom_df$chr,
          ranges   = IRanges(start = prom_df$start, end = prom_df$end)
        )
        names(cre_gr) <- prom_df$peak
        cre_gr <- unique(cre_gr)

        tx_all <- transcripts(EnsDb.Hsapiens.v86, filter = GeneNameFilter(unique(prom_df$gene)))
        tx_df  <- as.data.frame(tx_all) %>%
          mutate(
            chr = paste0("chr", as.character(seqnames)),
            chr = ifelse(chr == "chrMT", "chrM", chr),
            tss = ifelse(strand == "+", start, end)
          ) %>%
          filter(chr %in% c(paste0("chr", 1:22), "chrX", "chrY")) %>%
          select(gene = gene_name, chr, strand, tss)

        prom_df <- prom_df %>% mutate(cre_center = (start + end) / 2)

        links_out <- prom_df %>%
          inner_join(tx_df, by = c("gene", "chr"), relationship = "many-to-many") %>%
          mutate(distance_bp = abs(cre_center - tss)) %>%
          group_by(peak, gene) %>%
          slice_min(order_by = distance_bp, n = 1, with_ties = FALSE) %>%
          ungroup() %>%
          mutate(
            peak_type       = "promoter",
            distance_scaled = as.numeric(scale(log1p(distance_bp)))
          ) %>%
          select(peak, gene, distance_bp, distance_scaled, peak_type)

        message("[grn_exports] RNA_ONLY CRE->Gene edges: ", nrow(links_out),
                " | unique CREs=", length(unique(links_out$peak)),
                " | unique genes=", length(unique(links_out$gene)))
      }

      # ----------------------------------------------------------
      # Step B: motif matching on CRE intervals => TF->CRE edges
      # ----------------------------------------------------------
      message("[grn_exports] Motif matching on CREs (JASPAR2020) ... this can take a while.")

      pfm <- TFBSTools::getMatrixSet(
        JASPAR2020,
        opts = list(species = jaspar_species, all_versions = jaspar_all_versions)
      )
      motif_map <- data.frame(
        motif = names(pfm),
        tf    = vapply(pfm, function(x) gsub("\\s*\\(var\\..*\\)$", "", x@name), character(1L)),
        stringsAsFactors = FALSE
      )

      hvg_tfs <- intersect(hvg, motif_map$tf)
      pfm_hvg <- pfm[motif_map$motif[motif_map$tf %in% hvg_tfs]]


      motif_scores <- motifmatchr::matchMotifs(
        pwms    = pfm_hvg,
        subject = cre_gr,
        genome  = BSgenome.Hsapiens.UCSC.hg38,
        out     = "scores"
      )

      score_matrix <- motifScores(motif_scores)

      motif_long <- as.data.frame(summary(score_matrix)) %>%
        mutate(
          peak  = rownames(score_matrix)[i],
          motif = colnames(score_matrix)[j],
          score = x
        )

      tf_peak_scores <- motif_long %>%
        left_join(motif_map, by = "motif") %>%
        filter(tf %in% hvg) %>%
        group_by(tf, peak) %>%
        summarise(raw_score = max(score, na.rm = TRUE), .groups = "drop") %>%
        filter(raw_score > motif_raw_threshold) %>%
        group_by(tf) %>%
        mutate(score_z = as.numeric(scale(raw_score))) %>%
        ungroup() %>%
        select(tf, peak, raw_score, score_z)

      message("[grn_exports] TF->CRE edges after filtering: ", nrow(tf_peak_scores),
              " (raw_score > ", motif_raw_threshold, ")")

      # ----------------------------------------------------------
      # Step C: write CSVs 
      # ----------------------------------------------------------
      peaks_path <- file.path(self$csv_dir, peaks_genes_filename)
      tfpk_path  <- file.path(self$csv_dir, tf_peak_scores_filename)

      write.csv(links_out, peaks_path, row.names = FALSE)
      write.csv(tf_peak_scores, tfpk_path, row.names = FALSE)

      message("\n[grn_exports] Export complete:")
      message("  - CRE->Gene: ", normalizePath(peaks_path))
      message("  - TF->CRE:   ", normalizePath(tfpk_path))

      
      # ----------------------------------------------------------
      # Step D: Export feature matrices
      # ----------------------------------------------------------
      message("\n[grn_exports] Exporting feature matrices...")

      raw_dir <- file.path(self$feature_mtx_dir, "raw")
      proc_dir <- file.path(self$feature_mtx_dir, "processed")
      dir.create(raw_dir, showWarnings = FALSE, recursive = TRUE)
      dir.create(proc_dir, showWarnings = FALSE, recursive = TRUE)

      peaks_genes_df <- read.csv(peaks_path)
      tf_peaks_df <- read.csv(tfpk_path)

      tfs <- unique(tf_peaks_df$tf)
      target_genes <- unique(peaks_genes_df$gene)
      cres <- unique(peaks_genes_df$peak)
      barcodes <- colnames(self$obj)

      rna_genes <- rownames(private$get_layer(self$obj, "RNA", "counts"))
      sct_genes <- rownames(private$get_layer(self$obj, "SCT", "data"))
      tfs_final <- intersect(tfs, intersect(rna_genes, sct_genes))
      genes_final <- intersect(target_genes, sct_genes)

      if (self$mode == "MULTIOME") {
        message("Features: TFs=", length(tfs_final),
          " Genes=", length(genes_final),
          " CREs=", length(cres))
      } else {
        message("Features: TFs=", length(tfs_final),
          " Genes=", length(genes_final),
          " (RNA-only, no CRE features)")
      }

      rna_counts <- private$get_layer(self$obj, "RNA", "counts")
      sct_norm <- private$get_layer(self$obj, "SCT", "data")

      tf_rna <- rna_counts[tfs_final, barcodes, drop = FALSE]
      gene_rna <- rna_counts[genes_final, barcodes, drop = FALSE]
      writeMM(tf_rna, file.path(raw_dir, "tf_raw.mtx"))
      writeMM(gene_rna, file.path(raw_dir, "gene_raw.mtx"))

      tf_sct <- sct_norm[tfs_final, barcodes, drop = FALSE]
      gene_sct <- sct_norm[genes_final, barcodes, drop = FALSE]
      writeMM(tf_sct, file.path(proc_dir, "tf_sct.mtx"))
      writeMM(gene_sct, file.path(proc_dir, "gene_sct.mtx"))

      if (self$mode == "MULTIOME") {
        message("  [MULTIOME] Real ATAC peaks...")
        atac_counts <- private$get_layer(self$obj, "ATAC", "counts")
        atac_norm <- private$get_layer(self$obj, "ATAC", "data")
        cres_final <- intersect(cres, rownames(atac_counts))
  
        cre_raw <- atac_counts[cres_final, barcodes, drop = FALSE]
        cre_norm <- atac_norm[cres_final, barcodes, drop = FALSE]
  
        writeMM(cre_raw, file.path(raw_dir, "cre_counts_raw.mtx"))
        writeMM(cre_norm, file.path(proc_dir, "cre_counts.mtx"))

        writeLines(cres_final, file.path(self$feature_mtx_dir, "cre_features.txt"))
  
      } else {
        message("  [RNA_ONLY] Skipping CRE features (RNA-only mode).")
      }

      writeLines(barcodes, file.path(self$feature_mtx_dir, "barcodes.txt"))
      writeLines(tfs_final, file.path(self$feature_mtx_dir, "tf_features.txt"))
      writeLines(genes_final, file.path(self$feature_mtx_dir, "gene_features.txt"))

      message(" Feature matrices complete!")
      if (self$mode == "MULTIOME") {
        message("   raw/ → tf_raw.mtx, gene_raw.mtx, cre_counts_raw.mtx")
        message("   processed/ → tf_sct.mtx, gene_sct.mtx, cre_counts.mtx")
      } else {
        message("   raw/ → tf_raw.mtx, gene_raw.mtx")
        message("   processed/ → tf_sct.mtx, gene_sct.mtx")
      }


      self$obj <- obj
      invisible(self)
    }
    
  ),

    private = list(
      validate_inputs = function() {
        if (is.null(self$mode) || is.null(self$source)) stop("[validate_inputs] mode/source not set.")

        if (self$source == "10X_H5") {
          if (is.null(self$tenx_h5_path) || !nzchar(self$tenx_h5_path)) {
            stop("[validate_inputs] source=10X_H5 requires tenx_h5_path.")
          }
          if (self$mode == "MULTIOME" && (is.null(self$tenx_frag_path) || !nzchar(self$tenx_frag_path))) {
            stop("[validate_inputs] MULTIOME + 10X_H5 requires tenx_frag_path.")
          }
        }

        if (self$source == "CUSTOM") {
          if (is.null(self$custom_rna_path) || !nzchar(self$custom_rna_path)) {
            stop("[validate_inputs] source=CUSTOM requires custom_rna_path (RDS matrix).")
          }
          if (self$mode == "MULTIOME") {
            if (is.null(self$custom_atac_path) || !nzchar(self$custom_atac_path)) {
              stop("[validate_inputs] MULTIOME + CUSTOM requires custom_atac_path (RDS matrix).")
            }
            if (is.null(self$custom_frag_path) || !nzchar(self$custom_frag_path)) {
              stop("[validate_inputs] MULTIOME + CUSTOM requires custom_frag_path.")
            }
          }
        }

        if (self$source == "SEURAT_OBJECT") {
        }

        invisible(TRUE)
      },

      save_plot_if = function(p, filename, width, height, dpi) {
        out_path <- file.path(self$plot_dir, filename)
        ggplot2::ggsave(
          filename = out_path,
          plot = p,
          width = width,
          height = height,
          dpi = dpi
        )
        invisible(out_path)
      },

      get_layer = function(obj, assay = "RNA", layer = "counts") {
        out <- tryCatch({
          SeuratObject::LayerData(object = obj, assay = assay, layer = layer)
        }, error = function(e) NULL)

        if (!is.null(out)) return(out)

        a <- obj[[assay]]
        if (methods::is(a, "Assay")) {
          if (layer == "counts" && !is.null(a@counts)) return(a@counts)
          if (layer == "data"   && !is.null(a@data))   return(a@data)
          if (layer == "scale.data" && !is.null(a@scale.data)) return(a@scale.data)
        }

        out2 <- tryCatch({
          SeuratObject::GetAssayData(object = obj, assay = assay, layer = layer)
        }, error = function(e) NULL)

        if (!is.null(out2)) return(out2)

        stop("[get_layer] Could not retrieve layer='", layer, "' from assay='", assay,
            "'. Available layers: ",
            paste(tryCatch(SeuratObject::Layers(obj[[assay]]), error=function(e) character(0)), collapse = ", "))
      },

      peaks_to_granges = function(peaks_chr_colon) {

        df <- tibble::tibble(peak = peaks_chr_colon) %>%
          mutate(peak_norm = gsub(":", "-", peak)) %>%
          tidyr::separate(peak_norm, into = c("chr","start","end"), sep = "-", convert = TRUE, remove = FALSE)

        gr <- GRanges(seqnames = df$chr, ranges = IRanges(start = df$start, end = df$end))
        names(gr) <- df$peak
        unique(gr)
      },

      make_promoter_cre = function(obj, promoter_window_bp = 2000) {
        hvg <- VariableFeatures(obj, assay = "SCT")
        if (length(hvg) == 0) {
          warning("[make_promoter_cre] No SCT variable features found. Using all genes.")
        hvg <- rownames(obj[["RNA"]])
      }
  
        genes_present <- intersect(rownames(obj[["RNA"]]), hvg)
        tx_all <- transcripts(EnsDb.Hsapiens.v86, filter = GeneNameFilter(unique(genes_present)))

        tx_df <- as.data.frame(tx_all) %>%
          mutate(
            chr = paste0("chr", as.character(seqnames)),
            chr = ifelse(chr == "chrMT", "chrM", chr),
            tss = ifelse(strand == "+", start, end)
          ) %>%
          filter(chr %in% c(paste0("chr", 1:22), "chrX", "chrY")) %>%


          transmute(
            gene = gene_name,
            chr  = chr,
            start = pmax(1L, as.integer(tss - promoter_window_bp)),
            end   = as.integer(tss + promoter_window_bp),
            peak = paste0(chr, ":", start, "-", end)
          )


        tx_df
      }
    )          
  )


