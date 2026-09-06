export type Band = "low" | "medium" | "high";
export type Level = "L1" | "L2";

export interface FeatureDef {
  name: string;
  label: string;
  group: string;
  kind: "scale" | "int" | "choice";
  min?: number;
  max?: number;
  hint?: string | null;
  values: Record<string, string>;
  default: string | number;
  self_label?: string | null;
  self_hint?: string | null;
}

export interface GroupDef {
  id: string;
  label: string;
  hint: string;
}

export interface SurveyStep {
  id: string;
  title: string;
  features: string[];
  optional?: boolean;
}

export interface Schema {
  groups: GroupDef[];
  features: FeatureDef[];
  survey_features: string[];
  survey_steps: SurveyStep[];
  subject: string;
  survey_core: string[];
  survey_grades: string[];
  survey_steps_short: SurveyStep[];
  required_l2: string[];
  grade_features: string[];
}

export interface Factor {
  feature: string;
  label: string;
  value: string | number | null;
  value_label: string;
  effect: number;
  direction: "up" | "down";
  note: string;
  not_a_lever: boolean;
  weak?: boolean;
  unusual?: boolean;
}

export interface Row {
  id: string;
  row: number;
  probability: number;
  risk_band: Band;
  rank: number;
  in_shortlist: boolean;
  features: Record<string, string | number | null>;
  top_factors: Factor[];
  actual?: { G3: number; no_pass: boolean };
}

export interface Summary {
  n_rows: number;
  mean_probability: number;
  median_probability: number;
  in_shortlist: number;
  bands: Record<Band, number>;
  histogram: { counts: number[]; edges: number[] };
  elapsed_ms: number;
}

export interface Verification {
  n: number;
  n_fail?: number;
  roc_auc?: number;
  pr_auc?: number;
  confusion?: { tn: number; fp: number; fn: number; tp: number };
  precision?: number | null;
  recall?: number | null;
  note?: string;
  in_sample?: boolean;
}

export interface BatchResult {
  level: Level;
  level_note: string;
  threshold: number;
  default_threshold: number;
  capacity: number;
  risk_bands: { low_max: number; high_min: number; note: string };
  band_labels: Record<Band, string>;
  file: {
    filename: string;
    format?: string;
    delimiter?: string | null;
    n_rows?: number;
    n_cols?: number;
    demo?: boolean;
  };
  data_quality: {
    rows_with_missing: number;
    columns_with_missing: Record<string, number>;
    note: string;
  };
  summary: Summary;
  rows: Row[];
  id_column: string | null;
  privacy: string;
  verification?: Verification;
}

export interface SurveyResult {
  probability: number;
  risk_band: Band;
  band_label: string;
  top_factors: Factor[];
  comparison: {
    feature: string;
    label: string;
    you: number;
    typical: number;
    you_label: string;
    typical_label: string;
  }[];
  cohort: { n: number; fail_rate: number };
  threshold: number;
  level: Level;
  level_note: string;
  defaults_used: number;
  disclaimer: Record<string, string>;
}

export interface ModelInfo {
  trained_at: string;
  versions: Record<string, string>;
  training_data: {
    file: string;
    n_rows: number;
    n_positive: number;
    positive_rate: number;
    sha256: string;
  };
  metrics: Record<Level, {
    roc_auc: number;
    roc_auc_std: number;
    pr_auc: number;
    pr_auc_std: number;
  }>;
  threshold: { default: number; capacity: number; caught_at_capacity: number; note: string };
  risk_bands: { low_max: number; high_min: number; note: string };
  drop_features: string[];
  ensemble_members: Record<string, string[]>;
  importances: Record<Level, {
    feature: string;
    importance: number;
    direction: { kind: string; text: string; sign?: number };
  }[]>;
  levels: Record<Level, { name: string; uses: string; caveat: string }>;
}

export interface ApiError {
  message: string;
  hint?: string;
  details?: Record<string, unknown>;
}
