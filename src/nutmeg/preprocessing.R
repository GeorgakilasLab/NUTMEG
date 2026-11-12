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
  library(tibble)
})

GRNPrep <- R6Class(
  "GRNPrep",
  lock_objects = FALSE,
  public = list(
    mode = NULL,
    source = NULL,
    experiment = NULL,
    base_dir = NULL,
    tenx_h5_path = NULL,
    tenx_frag_path = NULL,
    rds_file = NULL,
    obj = NULL,
    annotation = NULL,
    seed = NULL,
    csv_dir = NULL,
    plot_dir = NULL,
    rds_dir = NULL,
    plots = NULL,

    initialize = function(
      mode = c("RNA_ONLY", "MULTIOME"),
      source = c("10X_H5", "SEURAT_OBJECT"),
      experiment = "exp_01",
      tenx_h5_path = NULL,
      tenx_frag_path = NULL,
      rds_file = NULL,
      seed = 42
    ) {
      self$mode <- match.arg(mode)
      self$source <- match.arg(source)
      self$experiment <- experiment
      self$base_dir <- experiment
      self$tenx_h5_path <- tenx_h5_path
      self$tenx_frag_path <- tenx_frag_path
      self$rds_file <- rds_file
      self$seed <- seed
      self$obj <- NULL
      self$plots <- list()

      set.seed(self$seed)
      private$validate_inputs()
      private$create_output_dirs()

      message(
        "\n[GRNPrep] Initialized\n",
        " - experiment: ", self$experiment, "\n",
        " - mode: ", self$mode, "\n",
        " - source: ", self$source, "\n",
        " - seed: ", self$seed, "\n",
        " - csv_dir: ", normalizePath(self$csv_dir), "\n",
        " - plot_dir: ", normalizePath(self$plot_dir), "\n",
        " - rds_dir: ", normalizePath(self$rds_dir), "\n"
      )

      invisible(self)
    },

    load = function(keep_standard_chromosomes = TRUE) {
      set.seed(self$seed)
      message(
        "\n[load] Loading data... (source=", self$source,
        ", mode=", self$mode, ")"
      )

      self$annotation <- GetGRangesFromEnsDb(
        ensdb = EnsDb.Hsapiens.v86
      )
      seqlevelsStyle(self$annotation) <- "UCSC"

      obj <- switch(
        self$source,
        "10X_H5" = private$load_from_10x_h5(
          keep_standard_chromosomes = keep_standard_chromosomes
        ),
        "SEURAT_OBJECT" = private$load_from_seurat_object(),
        stop("[load] Unsupported source: ", self$source)
      )

      self$obj <- obj
      message("[load] Done. Cells=", ncol(self$obj))
      invisible(self)
    },

    qc = function(
      save_plots = TRUE,
      filename = "qc_hist_before_filtering.png",
      force = FALSE
    ) {
      if (is.null(self$obj)) self$load()

      obj <- self$obj
      md <- obj@meta.data

      if (
        !force &&
        all(c("log_nCount_RNA", "percent.mt") %in% colnames(md))
      ) {
        message("[qc] QC metrics already present; skipping qc().")
        return(invisible(self$plots$qc_before_filtering))
      }

      message("\n[qc] Computing QC metrics... (mode=", self$mode, ")")

      DefaultAssay(obj) <- "RNA"
      obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
      obj$log_nCount_RNA <- log10(obj$nCount_RNA)

      hist_list <- list(
        private$make_histogram(
          obj@meta.data,
          "log_nCount_RNA",
          "log10(nCount_RNA)"
        ),
        private$make_histogram(
          obj@meta.data,
          "percent.mt",
          "percent.mt"
        )
      )

      if (self$mode == "MULTIOME") {
        DefaultAssay(obj) <- "ATAC"
        obj <- NucleosomeSignal(obj)
        obj <- TSSEnrichment(obj)

        obj$log_nCount_ATAC <- log10(obj$nCount_ATAC)
        obj$log_TSS.enrichment <- log10(obj$TSS.enrichment)
        obj$log_nucleosome_signal <- log10(obj$nucleosome_signal)

        hist_list <- c(
          hist_list,
          list(
            private$make_histogram(
              obj@meta.data,
              "log_nCount_ATAC",
              "log10(nCount_ATAC)"
            ),
            private$make_histogram(
              obj@meta.data,
              "log_TSS.enrichment",
              "log10(TSS enrichment)"
            ),
            private$make_histogram(
              obj@meta.data,
              "log_nucleosome_signal",
              "log10(Nucleosome signal)"
            )
          )
        )
      }

      qc_plot <- wrap_plots(hist_list, ncol = 3) +
        plot_annotation(
          title = if (self$mode == "MULTIOME") {
            "QC metrics (Multiome) before filtering"
          } else {
            "QC metrics (RNA-only) before filtering"
          },
          theme = theme(
            plot.title = element_text(
              size = 18,
              face = "bold",
              hjust = 0.5
            )
          )
        )

      self$obj <- obj
      self$plots$qc_before_filtering <- qc_plot

      if (save_plots) {
        private$save_plot(
          qc_plot,
          filename,
          width = 14,
          height = 10,
          dpi = 300
        )

        message(
          "[qc] Done. Plot saved -> ",
          file.path(self$plot_dir, filename)
        )
      } else {
        message("[qc] Done.")
      }

      invisible(qc_plot)
    },

    filter = function(
      rna_log_ncount_min = 3,
      rna_log_ncount_max = 4,
      rna_mt_pct_max = 20,
      atac_log_ncount_min = 3,
      atac_log_ncount_max = 4.5,
      atac_log_nucleo_max = 0.3,
      atac_log_tss_min = 0.0,
      min_cells_per_gene = 3,
      min_cells_per_peak = 10
    ) {
      if (is.null(self$obj)) {
        stop("[filter] Run $load() and $qc() before filtering.")
      }

      obj <- self$obj
      md <- obj@meta.data
      message(
        "\n[filter] Filtering cells & features... (mode=", self$mode, ")"
      )

      private$require_columns(
        md,
        c("log_nCount_RNA", "percent.mt"),
        "[filter] Missing RNA QC metrics. Run $qc() first."
      )

      if (self$mode == "MULTIOME") {
        private$require_columns(
          md,
          c(
            "log_nCount_ATAC",
            "log_TSS.enrichment",
            "log_nucleosome_signal"
          ),
          "[filter] Missing ATAC QC metrics. Run $qc() first."
        )
      }

      n_cells_before <- ncol(obj)

      keep_cells <- private$cell_filter_mask(
        md = md,
        mode = self$mode,
        rna_log_ncount_min = rna_log_ncount_min,
        rna_log_ncount_max = rna_log_ncount_max,
        rna_mt_pct_max = rna_mt_pct_max,
        atac_log_ncount_min = atac_log_ncount_min,
        atac_log_ncount_max = atac_log_ncount_max,
        atac_log_nucleo_max = atac_log_nucleo_max,
        atac_log_tss_min = atac_log_tss_min
      )

      obj <- obj[, keep_cells]

      obj[["RNA"]] <- subset(
        obj[["RNA"]],
        features = private$features_detected_in_cells(
          obj,
          "RNA",
          min_cells_per_gene
        )
      )

      if (self$mode == "MULTIOME") {
        obj[["ATAC"]] <- subset(
          obj[["ATAC"]],
          features = private$features_detected_in_cells(
            obj,
            "ATAC",
            min_cells_per_peak
          )
        )
      }

      self$obj <- obj
      message(
        "[filter] Done. Cells: ",
        n_cells_before, " -> ", ncol(obj)
      )

      invisible(self)
    },

    preprocess = function(
      n_sct_features = 2000,
      n_pca_dims = 50,
      n_lsi_dims = 20,
      resolution = 0.5,
      save_plots = TRUE,
      label_mode = c("clusters", "labels"),
      labels_column_name = "cell_label",
      labels_file = NULL,
      labels_sep = ",",
      final_rds_filename = NULL
    ) {
      if (is.null(self$obj)) {
        stop(
          "[preprocess] Run $load(), $qc(), and $filter() first."
        )
      }

      obj <- self$obj
      label_mode <- match.arg(label_mode)
      set.seed(self$seed)

      message(
        "\n[preprocess] Preprocessing + labeling (mode=", self$mode,
        ", labels=", label_mode, ")"
      )

      obj <- private$run_rna_preprocessing(
        obj = obj,
        n_sct_features = n_sct_features,
        n_pca_dims = n_pca_dims
      )

      if (self$mode == "RNA_ONLY") {
        obj <- private$run_rna_clustering(
          obj = obj,
          n_pca_dims = n_pca_dims,
          resolution = resolution
        )

        plot_name <- "umap_rna_only_clusters.png"
      } else {
        obj <- private$run_multiome_preprocessing(
          obj = obj,
          n_pca_dims = n_pca_dims,
          n_lsi_dims = n_lsi_dims,
          resolution = resolution
        )

        plot_name <- "umap_wnn_clusters.png"

        cluster_plot <- DimPlot(
          obj,
          group.by = "seurat_clusters",
          label = TRUE,
          repel = TRUE,
          reduction = "umap"
        ) + NoLegend()

        self$plots$umap_clusters <- cluster_plot

        if (save_plots) {
          private$save_plot(
            cluster_plot,
            plot_name,
            width = 8,
            height = 6,
            dpi = 300
          )
        }
      }

      obj <- private$assign_labels(
        obj = obj,
        label_mode = label_mode,
        labels_column_name = labels_column_name,
        labels_file = labels_file,
        labels_sep = labels_sep
      )

      private$write_labels_csv(obj, labels_column_name)
      private$write_final_rds(obj, final_rds_filename)

      self$obj <- obj
      invisible(self)
    },

    grn_exports = function(
      motif_raw_threshold = 5,
      jaspar_species = "Homo sapiens",
      jaspar_all_versions = FALSE,
      promoter_window_bp = 2000,
      edge_filename = "edge.csv",
      node_filename = "node.csv",
      metadata_filename = "metadata.csv",
      labels_column_name = "cell_label",
      sample_id = NULL,
      sample_column = NULL,
      coordinate_columns = c(
        x_coord = NA_character_,
        y_coord = NA_character_,
        z_coord = NA_character_
      )
    ) {
      if (is.null(self$obj)) {
        stop(
          "[grn_exports] Run load/qc/filter/preprocess first ",
          "(or inject a preprocessed object)."
        )
      }

      obj <- self$obj
      set.seed(self$seed)

      message(
        "\n[grn_exports] Building unified NUTMEG inputs... (mode=",
        self$mode, ")"
      )

      if (!("SCT" %in% names(obj@assays))) {
        stop("[grn_exports] SCT assay not found. Run preprocess() first.")
      }

      hvg <- VariableFeatures(obj, assay = "SCT")
      message("[grn_exports] Gene-side set = ", length(hvg), " genes.")

      cre_gene_links <- if (self$mode == "MULTIOME") {
        private$build_multiome_cre_gene_links(
          obj = obj,
          hvg = hvg,
          promoter_window_bp = promoter_window_bp
        )
      } else {
        private$build_rna_only_cre_gene_links(
          obj = obj,
          promoter_window_bp = promoter_window_bp
        )
      }

      links_out <- cre_gene_links$links
      cre_gr <- cre_gene_links$cre_gr

      tf_peak_scores <- private$build_tf_peak_scores(
        cre_gr = cre_gr,
        hvg = hvg,
        motif_raw_threshold = motif_raw_threshold,
        jaspar_species = jaspar_species,
        jaspar_all_versions = jaspar_all_versions
      )

      private$write_unified_inputs(
        obj = obj,
        links_out = links_out,
        tf_peak_scores = tf_peak_scores,
        edge_filename = edge_filename,
        node_filename = node_filename,
        metadata_filename = metadata_filename,
        labels_column_name = labels_column_name,
        sample_id = sample_id,
        sample_column = sample_column,
        coordinate_columns = coordinate_columns
      )

      self$obj <- obj
      invisible(self)
    }
  ),

  private = list(
    validate_inputs = function() {
      if (self$source == "10X_H5") {
        if (
          is.null(self$tenx_h5_path) ||
          !nzchar(self$tenx_h5_path)
        ) {
          stop(
            "[validate_inputs] source='10X_H5' requires tenx_h5_path."
          )
        }

        if (
          self$mode == "MULTIOME" &&
          (
            is.null(self$tenx_frag_path) ||
            !nzchar(self$tenx_frag_path)
          )
        ) {
          stop(
            "[validate_inputs] MULTIOME + 10X_H5 requires ",
            "tenx_frag_path."
          )
        }
      }

      if (
        self$source == "SEURAT_OBJECT" &&
        (is.null(self$rds_file) || !nzchar(self$rds_file))
      ) {
        stop(
          "[validate_inputs] source='SEURAT_OBJECT' requires rds_file."
        )
      }

      invisible(TRUE)
    },

    create_output_dirs = function() {
      self$csv_dir <- file.path(self$base_dir, "interm", "csv")
      self$plot_dir <- file.path(self$base_dir, "interm", "plots")
      self$rds_dir <- file.path(self$base_dir, "rds")

      dirs <- c(self$csv_dir, self$plot_dir, self$rds_dir)
      invisible(lapply(
        dirs,
        dir.create,
        showWarnings = FALSE,
        recursive = TRUE
      ))
    },

    load_from_10x_h5 = function(keep_standard_chromosomes = TRUE) {
      if (!file.exists(self$tenx_h5_path)) {
        stop(
          "[load] tenx_h5_path not found: ",
          self$tenx_h5_path
        )
      }

      counts <- Read10X_h5(self$tenx_h5_path)
      if (!is.list(counts)) {
        message("[load] Single feature type detected; treating matrix as Gene Expression.")
        counts <- list("Gene Expression" = counts)
      }
      rna_counts <- counts$`Gene Expression`
      obj <- CreateSeuratObject(
        counts = rna_counts,
        assay = "RNA"
      )

      if (self$mode == "MULTIOME") {
        if (
          is.null(self$tenx_frag_path) ||
          !file.exists(self$tenx_frag_path)
        ) {
          stop("[load] MULTIOME requires a valid fragments path.")
        }

        peak_mat <- counts$Peaks

        if (keep_standard_chromosomes) {
          std_chr <- paste0("chr", c(1:22, "X", "Y"))
          keep_peaks <- grepl(
            paste0(
              "^(",
              paste(std_chr, collapse = "|"),
              "):"
            ),
            rownames(peak_mat)
          )

          peak_mat <- peak_mat[keep_peaks, , drop = FALSE]
        }

        obj[["ATAC"]] <- CreateChromatinAssay(
          counts = peak_mat,
          sep = c(":", "-"),
          fragments = self$tenx_frag_path,
          annotation = self$annotation
        )
      }

      obj
    },

    load_from_seurat_object = function() {
      if (!file.exists(self$rds_file)) {
        stop("[load] rds_file not found: ", self$rds_file)
      }

      obj <- readRDS(self$rds_file)

      if (!inherits(obj, "Seurat")) {
        stop("[load] rds_file does not contain a Seurat object.")
      }

      obj
    },

    make_histogram = function(md, column, title) {
      ggplot(md, aes(x = .data[[column]])) +
        geom_histogram(
          bins = 50,
          fill = "steelblue",
          color = "black"
        ) +
        theme_minimal(base_size = 14) +
        ggtitle(title) +
        theme(
          axis.title.y = element_blank(),
          axis.title.x = element_blank()
        )
    },

    save_plot = function(plot_obj, filename, width, height, dpi) {
      ggplot2::ggsave(
        filename = file.path(self$plot_dir, filename),
        plot = plot_obj,
        width = width,
        height = height,
        dpi = dpi
      )
    },

    require_columns = function(df, cols, error_message) {
      missing_cols <- setdiff(cols, colnames(df))

      if (length(missing_cols) > 0) {
        stop(error_message)
      }

      invisible(TRUE)
    },

    cell_filter_mask = function(
      md,
      mode,
      rna_log_ncount_min,
      rna_log_ncount_max,
      rna_mt_pct_max,
      atac_log_ncount_min,
      atac_log_ncount_max,
      atac_log_nucleo_max,
      atac_log_tss_min
    ) {
      keep <- if (mode == "RNA_ONLY") {
        with(
          md,
          log_nCount_RNA > rna_log_ncount_min &
            log_nCount_RNA < rna_log_ncount_max &
            percent.mt < rna_mt_pct_max
        )
      } else {
        with(
          md,
          log_nCount_RNA > rna_log_ncount_min &
            log_nCount_RNA < rna_log_ncount_max &
            percent.mt < rna_mt_pct_max &
            log_nCount_ATAC > atac_log_ncount_min &
            log_nCount_ATAC < atac_log_ncount_max &
            log_nucleosome_signal < atac_log_nucleo_max &
            log_TSS.enrichment > atac_log_tss_min
        )
      }

      !is.na(keep) & is.finite(keep) & keep
    },

    features_detected_in_cells = function(obj, assay, min_cells) {
      counts <- private$get_layer(
        obj,
        assay = assay,
        layer = "counts"
      )

      detected <- Matrix::rowSums(counts > 0)
      names(detected[detected >= min_cells])
    },

    run_rna_preprocessing = function(obj, n_sct_features, n_pca_dims) {
      if (!("SCT" %in% Assays(obj))) {
        DefaultAssay(obj) <- "RNA"

        message(
          "[preprocess] Running SCTransform (n_features=",
          n_sct_features, ") ..."
        )

        set.seed(self$seed)
        obj <- SCTransform(
          obj,
          variable.features.n = n_sct_features,
          seed.use = self$seed,
          verbose = FALSE
        )

        if (!("pca" %in% names(obj@reductions))) {
          message("[preprocess] Running PCA ...")
          set.seed(self$seed)
          obj <- RunPCA(
            obj,
            seed.use = self$seed,
            verbose = FALSE
          )
        } else {
          message(
            "[preprocess] SCT/PCA already present; ",
            "skipping available steps."
          )
        }
      }

      obj
    },

    run_rna_clustering = function(obj, n_pca_dims, resolution) {
      obj <- FindNeighbors(
        obj,
        reduction = "pca",
        dims = 1:n_pca_dims,
        graph.name = c("rna_nn", "rna_snn"),
        verbose = FALSE
      )

      set.seed(self$seed)
      obj <- RunUMAP(
        obj,
        reduction = "pca",
        dims = 1:n_pca_dims,
        metric = "correlation",
        seed.use = self$seed,
        verbose = FALSE
      )

      obj <- FindClusters(
        obj,
        resolution = resolution,
        graph.name = "rna_snn",
        verbose = FALSE
      )

      obj
    },

    run_multiome_preprocessing = function(
      obj,
      n_pca_dims,
      n_lsi_dims,
      resolution
    ) {
      if (!("ATAC" %in% Assays(obj))) {
        stop("[preprocess] MULTIOME requires ATAC assay.")
      }

      DefaultAssay(obj) <- "ATAC"

      if (!("lsi" %in% names(obj@reductions))) {
        obj <- FindTopFeatures(obj, min.cutoff = "q75")
        obj <- RunTFIDF(obj)
        obj <- RunSVD(obj, n = n_lsi_dims)
      } else {
        message(
          "[preprocess] LSI already present; skipping TFIDF/SVD."
        )
      }

      obj <- FindMultiModalNeighbors(
        object = obj,
        reduction.list = list("pca", "lsi"),
        dims.list = list(1:n_pca_dims, 2:n_lsi_dims),
        modality.weight.name = "RNA.weight",
        verbose = FALSE
      )

      set.seed(self$seed)
      obj <- RunUMAP(
        obj,
        nn.name = "weighted.nn",
        assay = "RNA",
        metric = "correlation",
        seed.use = self$seed,
        verbose = FALSE
      )

      obj <- FindClusters(
        obj,
        graph.name = "wsnn",
        algorithm = 4,
        resolution = resolution,
        verbose = FALSE
      )

      obj
    },

    assign_labels = function(
      obj,
      label_mode,
      labels_column_name,
      labels_file,
      labels_sep
    ) {
      message(
        "[preprocess] Adding labels (mode=", label_mode, ")"
      )

      if (label_mode == "clusters") {
        if (!"seurat_clusters" %in% colnames(obj@meta.data)) {
          stop("[preprocess] No clusters found.")
        }

        obj[[labels_column_name]] <- as.character(obj$seurat_clusters)
        Idents(obj) <- labels_column_name
        return(obj)
      }

      if (is.null(labels_file)) {
        stop("[preprocess] labels mode requires labels_file.")
      }

      if (!file.exists(labels_file)) {
        stop("[preprocess] labels_file not found: ", labels_file)
      }

      lab_df <- read.csv(
        labels_file,
        sep = labels_sep,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )

      if (ncol(lab_df) < 2) {
        stop(
          "[preprocess] labels_file needs at least 2 columns: ",
          "barcode,label."
        )
      }

      colnames(lab_df)[1:2] <- c("barcode", "label")

      common_bcs <- intersect(colnames(obj), lab_df$barcode)

      if (length(common_bcs) == 0) {
        stop(
          "[preprocess] No barcode overlap between object and ",
          "labels file."
        )
      }

      lab_df <- lab_df[
        match(common_bcs, lab_df$barcode),
        ,
        drop = FALSE
      ]

      obj <- obj[, common_bcs]
      obj[[labels_column_name]] <- lab_df$label
      Idents(obj) <- labels_column_name

      labeled_cells <- which(
        !is.na(obj[[labels_column_name]][, 1])
      )

      if (length(labeled_cells) == 0) {
        stop("[preprocess] No labeled cells found.")
      }

      message(
        "[preprocess] Subsetting: ",
        length(labeled_cells), "/", ncol(obj),
        " labeled cells"
      )

      obj[, labeled_cells]
    },

    write_labels_csv = function(obj, labels_column_name) {
      labels_df <- data.frame(
        barcode = Cells(obj),
        label = as.character(obj[[labels_column_name]][, 1]),
        stringsAsFactors = FALSE
      )

      labels_csv_path <- file.path(self$csv_dir, "labels.csv")
      write.csv(
        labels_df,
        labels_csv_path,
        row.names = FALSE,
        quote = FALSE
      )

      message(
        "[preprocess] labels.csv saved: ",
        normalizePath(labels_csv_path)
      )
    },

    write_final_rds = function(obj, final_rds_filename) {
      final_path <- file.path(self$rds_dir, final_rds_filename)
      saveRDS(obj, final_path)

      message(
        "[preprocess] Labeled object saved: ",
        normalizePath(final_path)
      )
    },

    build_multiome_cre_gene_links = function(
      obj,
      hvg,
      promoter_window_bp
    ) {
      if (!("ATAC" %in% names(obj@assays))) {
        stop("[grn_exports] MULTIOME requires ATAC assay.")
      }

      hvp <- VariableFeatures(obj, assay = "ATAC")
      message(
        "[grn_exports] Peak-side set = ",
        length(hvp), " peaks."
      )

      DefaultAssay(obj) <- "ATAC"

      main_chroms <- standardChromosomes(
        BSgenome.Hsapiens.UCSC.hg38
      )

      keep_peaks <- which(
        as.character(seqnames(granges(obj))) %in% main_chroms
      )

      obj[["ATAC"]] <- subset(
        obj[["ATAC"]],
        features = rownames(obj[["ATAC"]])[keep_peaks]
      )

      obj <- RegionStats(
        obj,
        genome = BSgenome.Hsapiens.UCSC.hg38
      )

      message(
        "[grn_exports] Running LinkPeaks (ATAC -> SCT) ..."
      )

      obj <- LinkPeaks(
        object = obj,
        peak.assay = "ATAC",
        expression.assay = "SCT",
        genes.use = hvg
      )

      links_df <- Links(obj) |> as.data.frame()
      links_df <- links_df %>% dplyr::filter(peak %in% hvp)

      if (nrow(links_df) == 0) {
        stop(
          "[grn_exports] No peak-gene links after filtering ",
          "to selected peaks."
        )
      }

      links_df <- links_df %>%
        mutate(peak_norm = gsub(":", "-", peak)) %>%
        separate(
          peak_norm,
          into = c("peak_chr", "peak_start", "peak_end"),
          sep = "-",
          convert = TRUE,
          remove = FALSE
        ) %>%
        mutate(peak_center = (peak_start + peak_end) / 2)

      tx_all <- transcripts(
        EnsDb.Hsapiens.v86,
        filter = GeneNameFilter(unique(links_df$gene))
      )

      tx_df <- as.data.frame(tx_all) %>%
        mutate(
          seqnames = paste0("chr", as.character(seqnames)),
          seqnames = ifelse(seqnames == "chrMT", "chrM", seqnames),
          tss = ifelse(strand == "+", start, end)
        ) %>%
        filter(
          seqnames %in% c(paste0("chr", 1:22), "chrX", "chrY")
        ) %>%
        select(tx_id, gene_name, seqnames, strand, tss)

      links_with_tss <- links_df %>%
        inner_join(
          tx_df,
          by = c("gene" = "gene_name", "peak_chr" = "seqnames"),
          relationship = "many-to-many"
        ) %>%
        mutate(distance_bp = abs(peak_center - tss))

      if (nrow(links_with_tss) == 0) {
        stop(
          "[grn_exports] No peak-gene links remained after joining ",
          "transcript TSS annotations."
        )
      }

      links_out <- links_with_tss %>%
        group_by(peak, gene) %>%
        slice_min(
          order_by = distance_bp,
          n = 1,
          with_ties = FALSE
        ) %>%
        ungroup() %>%
        mutate(
          peak_type = ifelse(
            distance_bp <= promoter_window_bp,
            "promoter",
            "distal"
          ),
          distance_scaled = as.numeric(scale(log1p(distance_bp)))
        ) %>%
        select(
          peak,
          gene,
          distance_bp,
          distance_scaled,
          peak_type,
          score,
          zscore,
          pvalue
        )

      cre_gr <- private$peaks_to_granges(
        unique(links_out$peak)
      )

      message(
        "[grn_exports] MULTIOME CRE->Gene edges: ",
        nrow(links_out),
        " | unique CREs=",
        length(unique(links_out$peak)),
        " | unique genes=",
        length(unique(links_out$gene))
      )

      list(links = links_out, cre_gr = cre_gr)
    },

    build_rna_only_cre_gene_links = function(
      obj,
      promoter_window_bp
    ) {
      message(
        "[grn_exports] RNA_ONLY: using promoter-only pseudo-CREs."
      )

      prom_df <- private$make_promoter_cre(
        obj = obj,
        promoter_window_bp = promoter_window_bp
      )

      required_cols <- c("chr", "start", "end", "gene", "peak")

      if (!all(required_cols %in% colnames(prom_df))) {
        stop(
          "[grn_exports] Internal error: promoter CRE table ",
          "missing required columns."
        )
      }

      cre_gr <- GRanges(
        seqnames = prom_df$chr,
        ranges = IRanges(
          start = prom_df$start,
          end = prom_df$end
        )
      )

      names(cre_gr) <- prom_df$peak
      cre_gr <- unique(cre_gr)

      tx_df <- private$get_transcript_tss(
        unique(prom_df$gene),
        gene_col = "gene_name",
        chr_col = "chr"
      )

      links_out <- prom_df %>%
        mutate(cre_center = (start + end) / 2) %>%
        inner_join(
          tx_df,
          by = c("gene" = "gene_name", "chr"),
          relationship = "many-to-many"
        ) %>%
        mutate(distance_bp = abs(cre_center - tss)) %>%
        group_by(peak, gene) %>%
        slice_min(
          order_by = distance_bp,
          n = 1,
          with_ties = FALSE
        ) %>%
        ungroup() %>%
        mutate(
          peak_type = "promoter",
          distance_scaled = as.numeric(scale(log1p(distance_bp)))
        ) %>%
        select(
          peak,
          gene,
          distance_bp,
          distance_scaled,
          peak_type
        )

      message(
        "[grn_exports] RNA_ONLY CRE->Gene edges: ",
        nrow(links_out),
        " | unique CREs=",
        length(unique(links_out$peak)),
        " | unique genes=",
        length(unique(links_out$gene))
      )

      list(links = links_out, cre_gr = cre_gr)
    },

    build_tf_peak_scores = function(
      cre_gr,
      hvg,
      motif_raw_threshold,
      jaspar_species,
      jaspar_all_versions
    ) {
      message(
        "[grn_exports] Motif matching on CREs (JASPAR2020) ... ",
        "this can take a while."
      )

      pfm <- TFBSTools::getMatrixSet(
        JASPAR2020,
        opts = list(
          species = jaspar_species,
          all_versions = jaspar_all_versions
        )
      )

      motif_map <- data.frame(
        motif = names(pfm),
        tf = vapply(
          pfm,
          function(x) {
            gsub("\\s*\\(var\\..*\\)$", "", x@name)
          },
          character(1L)
        ),
        stringsAsFactors = FALSE
      )

      hvg_tfs <- intersect(hvg, motif_map$tf)
      pfm_hvg <- pfm[motif_map$motif[motif_map$tf %in% hvg_tfs]]

      motif_scores <- motifmatchr::matchMotifs(
        pwms = pfm_hvg,
        subject = cre_gr,
        genome = BSgenome.Hsapiens.UCSC.hg38,
        out = "scores"
      )

      score_matrix <- motifScores(motif_scores)

      motif_long <- as.data.frame(summary(score_matrix)) %>%
        mutate(
          peak = rownames(score_matrix)[i],
          motif = colnames(score_matrix)[j],
          score = x
        )

      tf_peak_scores <- motif_long %>%
        left_join(motif_map, by = "motif") %>%
        filter(tf %in% hvg) %>%
        group_by(tf, peak) %>%
        summarise(
          raw_score = max(score, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        filter(raw_score > motif_raw_threshold) %>%
        group_by(tf) %>%
        mutate(score_z = as.numeric(scale(raw_score))) %>%
        ungroup() %>%
        select(tf, peak, raw_score, score_z)

      message(
        "[grn_exports] TF->CRE edges after filtering: ",
        nrow(tf_peak_scores),
        " (raw_score > ", motif_raw_threshold, ")"
      )

      tf_peak_scores
    },

    write_unified_inputs = function(
      obj,
      links_out,
      tf_peak_scores,
      edge_filename,
      node_filename,
      metadata_filename,
      labels_column_name,
      sample_id,
      sample_column,
      coordinate_columns
    ) {
      message("[grn_exports] Creating unified NUTMEG inputs...")

      cells <- colnames(obj)

      if (
        !length(cells) ||
        anyNA(cells) ||
        any(!nzchar(cells)) ||
        anyDuplicated(cells)
      ) {
        stop(
          "[grn_exports] Cell identifiers must be nonempty and unique."
        )
      }

      required_links <- c("peak", "gene")
      required_tf <- c("tf", "peak")

      if (
        !all(required_links %in% names(links_out)) ||
        !all(required_tf %in% names(tf_peak_scores))
      ) {
        stop(
          "[grn_exports] Relationship tables are missing ",
          "endpoint columns."
        )
      }

      pg <- unique(data.frame(
        peak = as.character(links_out$peak),
        gene = as.character(links_out$gene),
        stringsAsFactors = FALSE
      ))

      tp <- unique(data.frame(
        tf = as.character(tf_peak_scores$tf),
        peak = as.character(tf_peak_scores$peak),
        stringsAsFactors = FALSE
      ))

      if (
        anyNA(pg$peak) || any(!nzchar(pg$peak)) ||
        anyNA(pg$gene) || any(!nzchar(pg$gene)) ||
        anyNA(tp$tf) || any(!nzchar(tp$tf)) ||
        anyNA(tp$peak) || any(!nzchar(tp$peak))
      ) {
        stop(
          "[grn_exports] Relationship endpoints contain missing ",
          "or empty IDs."
        )
      }

      # Shared candidate edges.
      if (self$mode == "MULTIOME") {
        edges <- unique(rbind(
          data.frame(
            left_node = tp$tf,
            right_node = tp$peak,
            is_directional = rep(TRUE, nrow(tp))
          ),
          data.frame(
            left_node = pg$peak,
            right_node = pg$gene,
            is_directional = rep(TRUE, nrow(pg))
          )
        ))

        gene_ids <- sort(unique(c(tp$tf, pg$gene)))
        cre_ids <- sort(unique(c(tp$peak, pg$peak)))

        if (length(intersect(gene_ids, cre_ids))) {
          stop(
            "[grn_exports] Gene and CRE identifiers must not collide."
          )
        }
      } else {
        paths <- merge(tp, pg, by = "peak")

        edges <- unique(data.frame(
          left_node = paths$tf,
          right_node = paths$gene,
          is_directional = rep(TRUE, nrow(paths))
        ))

        gene_ids <- sort(
          unique(c(edges$left_node, edges$right_node))
        )

        cre_ids <- character()
      }

      if (!nrow(edges)) {
        stop("[grn_exports] No candidate edges were generated.")
      }

      # Metadata.
      md <- obj@meta.data[cells, , drop = FALSE]

      if (!is.null(sample_column)) {
        if (!sample_column %in% names(md)) {
          stop(
            "[grn_exports] sample_column not found: ",
            sample_column
          )
        }

        sample_values <- as.character(md[[sample_column]])
      } else {
        if (
          is.null(sample_id) ||
          length(sample_id) != 1L ||
          is.na(sample_id) ||
          !nzchar(as.character(sample_id))
        ) {
          stop("[grn_exports] Supply sample_id or sample_column.")
        }

        sample_values <- rep(
          as.character(sample_id),
          length(cells)
        )
      }

      if (anyNA(sample_values) || any(!nzchar(sample_values))) {
        stop(
          "[grn_exports] Sample identifiers cannot be missing ",
          "or empty."
        )
      }

      if (!is.null(labels_column_name)) {
        if (!labels_column_name %in% names(md)) {
          stop(
            "[grn_exports] Label column not found: ",
            labels_column_name
          )
        }

        labels <- as.character(md[[labels_column_name]])
      } else {
        labels <- rep(NA_character_, length(cells))
      }

      metadata <- data.frame(
        sample_id = sample_values,
        cell_id = cells,
        label = labels,
        x_coord = rep(NA_real_, length(cells)),
        y_coord = rep(NA_real_, length(cells)),
        z_coord = rep(NA_real_, length(cells)),
        stringsAsFactors = FALSE
      )

      axes <- c("x_coord", "y_coord", "z_coord")

      if (!all(axes %in% names(coordinate_columns))) {
        stop(
          "[grn_exports] coordinate_columns must name ",
          "x_coord, y_coord, z_coord."
        )
      }

      for (axis in axes) {
        source_col <- coordinate_columns[[axis]]

        if (!is.na(source_col) && nzchar(source_col)) {
          if (!source_col %in% names(md)) {
            stop(
              "[grn_exports] Coordinate column not found: ",
              source_col
            )
          }

          original <- md[[source_col]]
          values <- suppressWarnings(
            as.numeric(as.character(original))
          )

          if (
            any(!is.na(original) & is.na(values)) ||
            any(!is.na(values) & !is.finite(values))
          ) {
            stop(
              "[grn_exports] Invalid numeric coordinates in: ",
              source_col
            )
          }

          metadata[[axis]] <- values
        }
      }

      # RNA measurements.
      rna_raw <- private$get_layer(obj, "RNA", "counts")
      rna_norm <- private$get_layer(obj, "SCT", "data")

      available_genes <- Reduce(
        intersect,
        list(
          gene_ids,
          rownames(rna_raw),
          rownames(rna_norm)
        )
      )

      if (length(setdiff(gene_ids, available_genes))) {
        warning(
          "[grn_exports] Excluding ",
          length(setdiff(gene_ids, available_genes)),
          " genes without both raw and normalized measurements."
        )
      }

      gene_ids <- available_genes

      tf_ids <- unique(tp$tf)
      target_ids <- unique(pg$gene)

      gene_subtypes <- setNames(
        vapply(
          gene_ids,
          function(gene_id) {
            roles <- character()

            if (gene_id %in% tf_ids) {
              roles <- c(roles, "TF")
            }

            if (gene_id %in% target_ids) {
              roles <- c(roles, "target")
            }

            paste(roles, collapse = "|")
          },
          character(1)
        ),
        gene_ids
      )

      # ATAC measurements.
      if (self$mode == "MULTIOME") {
        atac_raw <- private$get_layer(obj, "ATAC", "counts")
        atac_norm <- private$get_layer(obj, "ATAC", "data")

        available_cres <- Reduce(
          intersect,
          list(
            cre_ids,
            rownames(atac_raw),
            rownames(atac_norm)
          )
        )

        if (length(setdiff(cre_ids, available_cres))) {
          warning(
            "[grn_exports] Excluding ",
            length(setdiff(cre_ids, available_cres)),
            " CREs without both raw and normalized measurements."
          )
        }

        cre_ids <- available_cres

        cre_subtypes <- setNames(
          vapply(
            cre_ids,
            function(peak_id) {
              types <- as.character(
                links_out$peak_type[
                  as.character(links_out$peak) == peak_id
                ]
              )

              types <- unique(
                tolower(trimws(types[!is.na(types)]))
              )

              # Consistent ordering.
              roles <- c("promoter", "distal")
              roles <- roles[roles %in% types]

              if (!length(roles)) {
                return(NA_character_)
              }

              paste(roles, collapse = "|")
            },
            character(1)
          ),
          cre_ids
        )
      } else {
        cre_subtypes <- setNames(character(), character())
      }

      # Output paths.
      dir.create(
        self$csv_dir,
        recursive = TRUE,
        showWarnings = FALSE
      )

      paths <- file.path(
        self$csv_dir,
        c(edge_filename, node_filename, metadata_filename)
      )

      if (anyDuplicated(paths)) {
        stop("[grn_exports] Output paths must be distinct.")
      }

      edge_path <- paths[1]
      node_path <- paths[2]
      metadata_path <- paths[3]

      # Write headers first.
      write.csv(
        data.frame(
          left_node = character(),
          right_node = character(),
          is_directional = logical()
        ),
        edge_path,
        row.names = FALSE,
        na = ""
      )

      write.csv(
        data.frame(
          node_id = character(),
          node_type = character(),
          node_subtype = character(),
          feature_value = numeric(),
          feature_name = character(),
          metadata_id = character()
        ),
        node_path,
        row.names = FALSE,
        na = ""
      )

      write.csv(
        metadata,
        metadata_path,
        row.names = FALSE,
        na = ""
      )

      exported_ids <- character()
      observation_count <- 0

      append_observations <- function(
        raw,
        normalized,
        ids,
        node_type,
        subtypes,
        feature_name
      ) {
        if (!length(ids)) {
          return(invisible(NULL))
        }

        if (
          !all(cells %in% colnames(raw)) ||
          !all(cells %in% colnames(normalized))
        ) {
          stop(
            "[grn_exports] Measurement matrices do not contain ",
            "all cells."
          )
        }

        for (cell in cells) {
          counts <- as.numeric(raw[ids, cell, drop = FALSE])

          if (any(!is.finite(counts)) || any(counts < 0)) {
            stop(
              "[grn_exports] Invalid raw measurements for cell: ",
              cell
            )
          }

          # A node is retained only when its raw value is > 0.
          keep <- which(counts > 0)

          if (!length(keep)) {
            next
          }

          retained <- ids[keep]

          values <- as.numeric(
            normalized[retained, cell, drop = FALSE]
          )

          if (any(!is.finite(values))) {
            stop(
              "[grn_exports] Non-finite normalized values for ",
              "cell: ", cell
            )
          }

          rows <- data.frame(
            node_id = retained,
            node_type = node_type,
            node_subtype = unname(subtypes[retained]),
            feature_value = values,
            feature_name = feature_name,
            metadata_id = cell,
            stringsAsFactors = FALSE
          )

          write.table(
            rows,
            file = node_path,
            sep = ",",
            append = TRUE,
            col.names = FALSE,
            row.names = FALSE,
            quote = TRUE,
            na = ""
          )

          exported_ids <<- union(exported_ids, retained)
          observation_count <<- observation_count + nrow(rows)
        }

        invisible(NULL)
      }

      append_observations(
        raw = rna_raw,
        normalized = rna_norm,
        ids = gene_ids,
        node_type = "gene",
        subtypes = gene_subtypes,
        feature_name = "rna_sct_data"
      )

      if (self$mode == "MULTIOME") {
        append_observations(
          raw = atac_raw,
          normalized = atac_norm,
          ids = cre_ids,
          node_type = "CRE",
          subtypes = cre_subtypes,
          feature_name = "atac_tfidf"
        )
      }

      if (!observation_count) {
        stop(
          "[grn_exports] No node observations passed the ",
          "raw-value filters."
        )
      }

      # Keep only candidate edges whose endpoints occur in at least one cell.
      edges <- edges[
        edges$left_node %in% exported_ids &
          edges$right_node %in% exported_ids,
        ,
        drop = FALSE
      ]

      write.csv(edges, edge_path, row.names = FALSE, na = "")

      message(
        "[grn_exports] Unified inputs complete:",
        "\n - Candidate edges: ", nrow(edges),
        "\n - Node observations: ", observation_count,
        "\n - Cells in metadata: ", nrow(metadata),
        "\n - Edge file: ", normalizePath(edge_path),
        "\n - Node file: ", normalizePath(node_path),
        "\n - Metadata file: ", normalizePath(metadata_path)
      )

      invisible(list(
        edge = edge_path,
        node = node_path,
        metadata = metadata_path
      ))
    },

    get_layer = function(obj, assay = "RNA", layer = "counts") {
      out <- tryCatch({
        SeuratObject::LayerData(
          object = obj,
          assay = assay,
          layer = layer
        )
      }, error = function(e) NULL)

      if (!is.null(out)) {
        return(out)
      }

      a <- obj[[assay]]

      if (methods::is(a, "Assay")) {
        if (layer == "counts" && !is.null(a@counts)) {
          return(a@counts)
        }

        if (layer == "data" && !is.null(a@data)) {
          return(a@data)
        }

        if (layer == "scale.data" && !is.null(a@scale.data)) {
          return(a@scale.data)
        }
      }

      out2 <- tryCatch({
        SeuratObject::GetAssayData(
          object = obj,
          assay = assay,
          layer = layer
        )
      }, error = function(e) NULL)

      if (!is.null(out2)) {
        return(out2)
      }

      stop(
        "[get_layer] Could not retrieve layer='", layer,
        "' from assay='", assay, "'. Available layers: ",
        paste(
          tryCatch(
            SeuratObject::Layers(obj[[assay]]),
            error = function(e) character(0)
          ),
          collapse = ", "
        )
      )
    },

    get_transcript_tss = function(
      genes,
      gene_col = "gene_name",
      chr_col = "seqnames"
    ) {
      tx_all <- transcripts(
        EnsDb.Hsapiens.v86,
        filter = GeneNameFilter(unique(genes))
      )

      tx_df <- as.data.frame(tx_all) %>%
        mutate(
          chr_tmp = paste0("chr", as.character(seqnames)),
          chr_tmp = ifelse(chr_tmp == "chrMT", "chrM", chr_tmp),
          tss = ifelse(strand == "+", start, end)
        ) %>%
        filter(
          chr_tmp %in% c(paste0("chr", 1:22), "chrX", "chrY")
        )

      names(tx_df)[names(tx_df) == "gene_name"] <- gene_col
      names(tx_df)[names(tx_df) == "chr_tmp"] <- chr_col

      tx_df[, c(gene_col, chr_col, "strand", "tss")]
    },

    peaks_to_granges = function(peaks_chr_colon) {
      df <- tibble(peak = peaks_chr_colon) %>%
        mutate(peak_norm = gsub(":", "-", peak)) %>%
        separate(
          peak_norm,
          into = c("chr", "start", "end"),
          sep = "-",
          convert = TRUE,
          remove = FALSE
        )

      gr <- GRanges(
        seqnames = df$chr,
        ranges = IRanges(start = df$start, end = df$end)
      )

      names(gr) <- df$peak
      unique(gr)
    },

    make_promoter_cre = function(obj, promoter_window_bp = 2000) {
      hvg <- VariableFeatures(obj, assay = "SCT")

      if (length(hvg) == 0) {
        warning(
          "[make_promoter_cre] No SCT variable features found. ",
          "Using all genes."
        )

        hvg <- rownames(obj[["RNA"]])
      }

      genes_present <- intersect(
        rownames(obj[["RNA"]]),
        hvg
      )

      tx_all <- transcripts(
        EnsDb.Hsapiens.v86,
        filter = GeneNameFilter(unique(genes_present))
      )

      as.data.frame(tx_all) %>%
        mutate(
          chr = paste0("chr", as.character(seqnames)),
          chr = ifelse(chr == "chrMT", "chrM", chr),
          tss = ifelse(strand == "+", start, end)
        ) %>%
        filter(
          chr %in% c(paste0("chr", 1:22), "chrX", "chrY")
        ) %>%
        transmute(
          gene = gene_name,
          chr = chr,
          start = pmax(
            1L,
            as.integer(tss - promoter_window_bp)
          ),
          end = as.integer(tss + promoter_window_bp),
          peak = paste0(chr, ":", start, "-", end)
        )
    }
  )
)