# Notes to the data card — tcga_luad_egfr

Added on 18/11/2026 (Y105). This file is kept apart from `data_card_tcga_luad_egfr.md`, whose SHA-256 is listed in
`B7_data/config.json` and checked by every later session; the data card itself is unchanged.

- The FILTER column of the mutation files was not stored at step 0; the FILTER values listed in the
  data card are therefore missing (`nan`). The labels are based on the open TCGA-LUAD Masked Somatic Mutation files as
  released by the GDC (Data Release 46.0), without further filtering on FILTER.
