use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use zombie_policy_core::{ACTION_COUNT, FEATURE_NAMES, PLAYER_FEATURE_NAMES};

const MODEL_TINY_Q_MLP_V1: &str = "tiny_q_mlp_v1";
const MODEL_TINY_Q_ATTENTION_V1: &str = "tiny_q_attention_v1";
const ZOMBIE_ACTION_NAMES: [&str; ACTION_COUNT] =
    ["ATTACK", "FLANK_LEFT", "FLANK_RIGHT", "DASH_IN", "DASH_OUT"];
const PLAYER_ACTION_NAMES: [&str; ACTION_COUNT] =
    ["ATTACK", "APPROACH", "RETREAT", "STRAFE", "ROLL"];
const MAX_Q_PARAMS: usize = 10_000;

#[derive(Debug, Clone)]
struct Config {
    input_path: String,
    output_path: String,
    model: String,
    role: String,
    input_dim: usize,
    hidden_dim: usize,
    d_model: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    actions: usize,
    epochs: usize,
    learning_rate: f32,
    gamma: f32,
    target_sync: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            output_path: String::new(),
            model: "mlp".to_string(),
            role: "zombie".to_string(),
            input_dim: 23,
            hidden_dim: 275,
            d_model: 40,
            layers: 2,
            heads: 4,
            kv_heads: 1,
            actions: ACTION_COUNT,
            epochs: 100,
            learning_rate: 0.006,
            gamma: 0.96,
            target_sync: 64,
        }
    }
}

#[derive(Debug, Clone)]
struct Transition {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
}

#[derive(Debug, Clone)]
struct FeatureStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TrainSummary {
    transitions: usize,
    epochs: usize,
    mean_abs_td: f32,
    parameter_count: usize,
}

#[derive(Debug, Clone)]
struct MlpPolicy {
    input_dim: usize,
    hidden_dim: usize,
    actions: usize,
    stats: FeatureStats,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    target_w1: Vec<Vec<f32>>,
    target_b1: Vec<f32>,
    target_w2: Vec<Vec<f32>>,
    target_b2: Vec<f32>,
    updates: usize,
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    rms_weight: Vec<f32>,
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct AttentionPolicy {
    input_dim: usize,
    d_model: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    actions: usize,
    stats: FeatureStats,
    feature_embedding: Vec<Vec<f32>>,
    value_embedding: Vec<f32>,
    q_token: Vec<f32>,
    transformer_layers: Vec<AttentionLayer>,
    final_rms_weight: Vec<f32>,
    q_head: Vec<Vec<f32>>,
    q_bias: Vec<f32>,
    target_q_head: Vec<Vec<f32>>,
    target_q_bias: Vec<f32>,
    updates: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    if config.input_path.is_empty() || config.output_path.is_empty() {
        print_usage();
        return Err("missing --input or --output".into());
    }
    if config.actions != ACTION_COUNT {
        return Err(format!("runtime checkpoint expects {ACTION_COUNT} actions").into());
    }

    let feature_names = feature_names_for_dim(config.input_dim)?;
    let transitions = read_transitions(&config.input_path, feature_names)?;
    if transitions.is_empty() {
        return Err(format!("no transitions found in {}", config.input_path).into());
    }
    let stats = compute_feature_stats(&transitions, config.input_dim);
    let action_names = action_names_for_role(&config.role);

    let summary = match config.model.as_str() {
        "mlp" => {
            let mut policy = MlpPolicy::new(config.input_dim, config.hidden_dim, config.actions, stats);
            let summary = policy.train(
                &transitions,
                config.epochs,
                config.learning_rate,
                config.gamma,
                config.target_sync,
            );
            ensure_parent_dir(&config.output_path)?;
            fs::write(&config.output_path, policy.to_json(action_names))?;
            summary
        }
        "attention" => {
            let mut policy = AttentionPolicy::new(
                config.input_dim,
                config.d_model,
                config.layers,
                config.heads,
                config.kv_heads,
                config.actions,
                stats,
            )?;
            let summary = policy.train(
                &transitions,
                config.epochs,
                config.learning_rate,
                config.gamma,
                config.target_sync,
            );
            ensure_parent_dir(&config.output_path)?;
            fs::write(&config.output_path, policy.to_json(action_names))?;
            summary
        }
        other => return Err(format!("unsupported --model {other}; use mlp or attention").into()),
    };

    println!(
        "Tiny Q offline trainer done: model={} role={} transitions={} epochs={} mean_abs_td={:.4} params={} -> {}",
        config.model,
        config.role,
        summary.transitions,
        summary.epochs,
        summary.mean_abs_td,
        summary.parameter_count,
        config.output_path
    );
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut config = Config::default();
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => config.input_path = take_value(&mut args, "--input")?,
            "--output" => config.output_path = take_value(&mut args, "--output")?,
            "--model" => config.model = take_value(&mut args, "--model")?.to_lowercase(),
            "--role" => config.role = take_value(&mut args, "--role")?.to_lowercase(),
            "--input-dim" => config.input_dim = take_value(&mut args, "--input-dim")?.parse()?,
            "--hidden-dim" => config.hidden_dim = take_value(&mut args, "--hidden-dim")?.parse()?,
            "--d-model" => config.d_model = take_value(&mut args, "--d-model")?.parse()?,
            "--layers" => config.layers = take_value(&mut args, "--layers")?.parse()?,
            "--heads" => config.heads = take_value(&mut args, "--heads")?.parse()?,
            "--kv-heads" => config.kv_heads = take_value(&mut args, "--kv-heads")?.parse()?,
            "--actions" => config.actions = take_value(&mut args, "--actions")?.parse()?,
            "--epochs" => config.epochs = take_value(&mut args, "--epochs")?.parse()?,
            "--lr" => config.learning_rate = take_value(&mut args, "--lr")?.parse()?,
            "--gamma" => config.gamma = take_value(&mut args, "--gamma")?.parse()?,
            "--target-sync" => config.target_sync = take_value(&mut args, "--target-sync")?.parse()?,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if config.input_dim == PLAYER_FEATURE_NAMES.len() && config.role == "zombie" {
        config.role = "player".to_string();
    }
    Ok(config)
}

fn take_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    name: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value after {name}").into())
}

fn print_usage() {
    eprintln!(
        "Usage: train_tiny_q --input <csv> --output <checkpoint.json> --model <mlp|attention> [--role zombie|player] [--epochs 100]"
    );
}

fn feature_names_for_dim(input_dim: usize) -> Result<&'static [&'static str], Box<dyn Error>> {
    if input_dim == FEATURE_NAMES.len() {
        Ok(&FEATURE_NAMES)
    } else if input_dim == PLAYER_FEATURE_NAMES.len() {
        Ok(&PLAYER_FEATURE_NAMES)
    } else {
        Err(format!("unsupported input dim {input_dim}; expected 23 zombie or 15 player features").into())
    }
}

fn action_names_for_role(role: &str) -> &'static [&'static str] {
    if role == "player" {
        &PLAYER_ACTION_NAMES
    } else {
        &ZOMBIE_ACTION_NAMES
    }
}

fn read_transitions(
    path: &str,
    feature_names: &[&str],
) -> Result<Vec<Transition>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let Some(header_line) = lines.next() else {
        return Ok(Vec::new());
    };
    let headers = parse_csv_line(header_line);
    let index_by_name = headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect::<HashMap<_, _>>();

    let action_idx = required_column(&index_by_name, "action_id")?;
    let reward_idx = required_column(&index_by_name, "reward")?;
    let done_idx = required_column(&index_by_name, "done")?;
    let prev_indices = feature_indices(&index_by_name, "prev_", feature_names)?;
    let next_indices = feature_indices(&index_by_name, "next_", feature_names)?;

    let mut transitions = Vec::new();
    for (line_number, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cells = parse_csv_line(line);
        let required_len = *prev_indices
            .iter()
            .chain(next_indices.iter())
            .chain([action_idx, reward_idx, done_idx].iter())
            .max()
            .unwrap_or(&0)
            + 1;
        if cells.len() < required_len {
            eprintln!("Skipping short CSV row {} in {}", line_number + 2, path);
            continue;
        }
        let action = cells[action_idx].parse::<usize>().unwrap_or(0);
        if action >= ACTION_COUNT {
            eprintln!("Skipping row {} with invalid action {}", line_number + 2, action);
            continue;
        }
        let reward = cells[reward_idx].parse::<f32>().unwrap_or(0.0);
        let done = parse_bool_cell(&cells[done_idx]);
        let state = prev_indices
            .iter()
            .map(|idx| cells[*idx].parse::<f32>().unwrap_or(0.0))
            .collect::<Vec<_>>();
        let next_state = next_indices
            .iter()
            .map(|idx| cells[*idx].parse::<f32>().unwrap_or(0.0))
            .collect::<Vec<_>>();
        transitions.push(Transition {
            state,
            action,
            reward,
            next_state,
            done,
        });
    }
    Ok(transitions)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                cells.push(current);
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    cells.push(current);
    cells
}

fn required_column(
    index_by_name: &HashMap<String, usize>,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    index_by_name
        .get(name)
        .copied()
        .ok_or_else(|| format!("missing required CSV column {name}").into())
}

fn feature_indices(
    index_by_name: &HashMap<String, usize>,
    prefix: &str,
    feature_names: &[&str],
) -> Result<Vec<usize>, Box<dyn Error>> {
    feature_names
        .iter()
        .map(|name| {
            let column = format!("{prefix}{name}");
            index_by_name
                .get(&column)
                .copied()
                .ok_or_else(|| format!("missing required CSV feature column {column}").into())
        })
        .collect()
}

fn parse_bool_cell(value: &str) -> bool {
    matches!(value.trim().to_lowercase().as_str(), "1" | "true" | "yes")
}

fn compute_feature_stats(transitions: &[Transition], input_dim: usize) -> FeatureStats {
    let mut mean = vec![0.0f32; input_dim];
    for transition in transitions {
        for (idx, value) in transition.state.iter().enumerate() {
            mean[idx] += *value;
        }
    }
    let count = transitions.len().max(1) as f32;
    for value in mean.iter_mut() {
        *value /= count;
    }

    let mut variance = vec![0.0f32; input_dim];
    for transition in transitions {
        for (idx, value) in transition.state.iter().enumerate() {
            let delta = *value - mean[idx];
            variance[idx] += delta * delta;
        }
    }
    let std = variance
        .into_iter()
        .map(|value| {
            let raw = (value / count).sqrt();
            if raw < 1.0e-4 { 1.0 } else { raw }
        })
        .collect();
    FeatureStats { mean, std }
}

impl MlpPolicy {
    fn new(input_dim: usize, hidden_dim: usize, actions: usize, stats: FeatureStats) -> Self {
        let input_scale = (input_dim as f32).sqrt().max(1.0);
        let hidden_scale = (hidden_dim as f32).sqrt().max(1.0);
        let mut w1 = vec![vec![0.0; input_dim]; hidden_dim];
        for (hidden_idx, row) in w1.iter_mut().enumerate() {
            for (input_idx, weight) in row.iter_mut().enumerate() {
                *weight = deterministic_weight((hidden_idx * input_dim + input_idx) as u64, input_scale);
            }
        }
        let b1 = vec![0.0; hidden_dim];
        let mut w2 = vec![vec![0.0; hidden_dim]; actions];
        for (action_idx, row) in w2.iter_mut().enumerate() {
            for (hidden_idx, weight) in row.iter_mut().enumerate() {
                *weight = deterministic_weight(
                    (10_000 + action_idx * hidden_dim + hidden_idx) as u64,
                    hidden_scale,
                );
            }
        }
        let b2 = vec![0.0; actions];
        Self {
            input_dim,
            hidden_dim,
            actions,
            stats,
            target_w1: w1.clone(),
            target_b1: b1.clone(),
            target_w2: w2.clone(),
            target_b2: b2.clone(),
            w1,
            b1,
            w2,
            b2,
            updates: 0,
        }
    }

    fn train(
        &mut self,
        transitions: &[Transition],
        epochs: usize,
        learning_rate: f32,
        gamma: f32,
        target_sync: usize,
    ) -> TrainSummary {
        let mut total_abs_td = 0.0f32;
        let mut update_count = 0usize;
        for _ in 0..epochs {
            for transition in transitions {
                total_abs_td += self.learn(transition, learning_rate, gamma, target_sync);
                update_count += 1;
            }
        }
        TrainSummary {
            transitions: transitions.len(),
            epochs,
            mean_abs_td: total_abs_td / update_count.max(1) as f32,
            parameter_count: self.parameter_count(),
        }
    }

    fn learn(
        &mut self,
        transition: &Transition,
        learning_rate: f32,
        gamma: f32,
        target_sync: usize,
    ) -> f32 {
        let lr = learning_rate.clamp(0.0001, 0.05);
        let gamma = gamma.clamp(0.0, 0.999);
        let reward = transition.reward.clamp(-10.0, 10.0);
        let normalized = self.normalize(&transition.state);
        let (hidden, q_values) = self.forward_from_normalized(&normalized, &self.w1, &self.b1, &self.w2, &self.b2);
        let (_, next_online) = self.forward(&transition.next_state);
        let next_normalized = self.normalize(&transition.next_state);
        let (_, next_target) = self.forward_from_normalized(
            &next_normalized,
            &self.target_w1,
            &self.target_b1,
            &self.target_w2,
            &self.target_b2,
        );
        let bootstrap = double_dqn_bootstrap_value(&next_online, &next_target, transition.done);
        let target = (reward + gamma * bootstrap).clamp(-10.0, 10.0);
        let td_error = (q_values[transition.action] - target).clamp(-4.0, 4.0);
        let old_w2_row = self.w2[transition.action].clone();

        self.b2[transition.action] -= lr * td_error;
        for hidden_idx in 0..self.hidden_dim {
            self.w2[transition.action][hidden_idx] -= lr * td_error * hidden[hidden_idx];
        }
        for hidden_idx in 0..self.hidden_dim {
            if hidden[hidden_idx] <= 0.0 {
                continue;
            }
            let hidden_grad = old_w2_row[hidden_idx] * td_error;
            self.b1[hidden_idx] -= lr * hidden_grad;
            for input_idx in 0..self.input_dim {
                self.w1[hidden_idx][input_idx] -= lr * hidden_grad * normalized[input_idx];
            }
        }

        self.updates += 1;
        if self.updates % target_sync.max(1) == 0 {
            self.sync_target();
        }
        td_error.abs()
    }

    fn forward(&self, features: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let normalized = self.normalize(features);
        self.forward_from_normalized(&normalized, &self.w1, &self.b1, &self.w2, &self.b2)
    }

    fn normalize(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value - self.stats.mean[idx]) / self.stats.std[idx].max(1.0e-6))
            .collect()
    }

    fn forward_from_normalized(
        &self,
        normalized: &[f32],
        w1: &[Vec<f32>],
        b1: &[f32],
        w2: &[Vec<f32>],
        b2: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = vec![0.0f32; self.hidden_dim];
        for hidden_idx in 0..self.hidden_dim {
            let mut sum = b1[hidden_idx];
            for input_idx in 0..self.input_dim {
                sum += w1[hidden_idx][input_idx] * normalized[input_idx];
            }
            hidden[hidden_idx] = sum.max(0.0);
        }

        let mut q_values = vec![0.0f32; self.actions];
        for action_idx in 0..self.actions {
            let mut sum = b2[action_idx];
            for hidden_idx in 0..self.hidden_dim {
                sum += w2[action_idx][hidden_idx] * hidden[hidden_idx];
            }
            q_values[action_idx] = sum;
        }
        (hidden, q_values)
    }

    fn sync_target(&mut self) {
        self.target_w1.clone_from(&self.w1);
        self.target_b1.clone_from(&self.b1);
        self.target_w2.clone_from(&self.w2);
        self.target_b2.clone_from(&self.b2);
    }

    fn parameter_count(&self) -> usize {
        self.hidden_dim * self.input_dim + self.hidden_dim + self.actions * self.hidden_dim + self.actions
    }

    fn to_json(&self, action_names: &[&str]) -> String {
        let mut output = String::new();
        writeln!(output, "{{").unwrap();
        writeln!(output, "  \"model_type\": \"{}\",", MODEL_TINY_Q_MLP_V1).unwrap();
        writeln!(output, "  \"input_dim\": {},", self.input_dim).unwrap();
        writeln!(output, "  \"hidden_dim\": {},", self.hidden_dim).unwrap();
        writeln!(output, "  \"num_actions\": {},", self.actions).unwrap();
        write_json_array_f32(&mut output, "feature_mean", &self.stats.mean, 2, true);
        write_json_array_f32(&mut output, "feature_std", &self.stats.std, 2, true);
        write_json_matrix_f32(&mut output, "w1", &self.w1, 2, true);
        write_json_array_f32(&mut output, "b1", &self.b1, 2, true);
        write_json_matrix_f32(&mut output, "w2", &self.w2, 2, true);
        write_json_array_f32(&mut output, "b2", &self.b2, 2, true);
        write_json_array_str(&mut output, "action_names", action_names, 2, false);
        writeln!(output, "}}").unwrap();
        output
    }
}

impl AttentionPolicy {
    fn new(
        input_dim: usize,
        d_model: usize,
        layers: usize,
        heads: usize,
        kv_heads: usize,
        actions: usize,
        stats: FeatureStats,
    ) -> Result<Self, Box<dyn Error>> {
        if d_model == 0 || heads == 0 || d_model % heads != 0 {
            return Err("attention d_model must be divisible by heads".into());
        }
        let kv_heads = normalize_kv_heads(kv_heads, heads);
        let parameter_count = attention_param_count(input_dim, d_model, layers, heads, kv_heads, actions);
        if parameter_count > MAX_Q_PARAMS {
            return Err(format!("attention parameter budget exceeded: {parameter_count} > {MAX_Q_PARAMS}").into());
        }

        let d_scale = (d_model as f32).sqrt().max(1.0);
        let head_dim = d_model / heads;
        let kv_dim = kv_heads * head_dim;
        let mut feature_embedding = vec![vec![0.0; d_model]; input_dim];
        for feature_idx in 0..input_dim {
            for dim in 0..d_model {
                feature_embedding[feature_idx][dim] =
                    deterministic_weight((1_000 + feature_idx * d_model + dim) as u64, d_scale);
            }
        }
        let mut value_embedding = vec![0.0; d_model];
        let mut q_token = vec![0.0; d_model];
        for dim in 0..d_model {
            value_embedding[dim] = deterministic_weight((5_000 + dim) as u64, d_scale);
            q_token[dim] = deterministic_weight((6_000 + dim) as u64, d_scale);
        }
        let mut transformer_layers = Vec::with_capacity(layers);
        for layer_idx in 0..layers {
            transformer_layers.push(AttentionLayer {
                rms_weight: vec![1.0; d_model],
                wq: deterministic_matrix(d_model, d_model, 10_000 + layer_idx as u64 * 10_000, d_scale),
                wk: deterministic_matrix(kv_dim, d_model, 20_000 + layer_idx as u64 * 10_000, d_scale),
                wv: deterministic_matrix(kv_dim, d_model, 30_000 + layer_idx as u64 * 10_000, d_scale),
                wo: deterministic_matrix(d_model, d_model, 40_000 + layer_idx as u64 * 10_000, d_scale),
            });
        }
        let final_rms_weight = vec![1.0; d_model];
        let mut q_head = vec![vec![0.0; d_model]; actions];
        for action_idx in 0..actions {
            for dim in 0..d_model {
                q_head[action_idx][dim] =
                    deterministic_weight((70_000 + action_idx * d_model + dim) as u64, d_scale);
            }
        }
        let q_bias = vec![0.0; actions];
        Ok(Self {
            input_dim,
            d_model,
            layers,
            heads,
            kv_heads,
            actions,
            stats,
            feature_embedding,
            value_embedding,
            q_token,
            transformer_layers,
            final_rms_weight,
            target_q_head: q_head.clone(),
            target_q_bias: q_bias.clone(),
            q_head,
            q_bias,
            updates: 0,
        })
    }

    fn train(
        &mut self,
        transitions: &[Transition],
        epochs: usize,
        learning_rate: f32,
        gamma: f32,
        target_sync: usize,
    ) -> TrainSummary {
        let mut total_abs_td = 0.0f32;
        let mut update_count = 0usize;
        for _ in 0..epochs {
            for transition in transitions {
                total_abs_td += self.learn(transition, learning_rate, gamma, target_sync);
                update_count += 1;
            }
        }
        TrainSummary {
            transitions: transitions.len(),
            epochs,
            mean_abs_td: total_abs_td / update_count.max(1) as f32,
            parameter_count: self.parameter_count(),
        }
    }

    fn learn(
        &mut self,
        transition: &Transition,
        learning_rate: f32,
        gamma: f32,
        target_sync: usize,
    ) -> f32 {
        let lr = learning_rate.clamp(0.0001, 0.05);
        let gamma = gamma.clamp(0.0, 0.999);
        let reward = transition.reward.clamp(-10.0, 10.0);
        let (q_repr, q_values) = self.forward(&transition.state, &self.q_head, &self.q_bias);
        let (_, next_online) = self.forward(&transition.next_state, &self.q_head, &self.q_bias);
        let (_, next_target) = self.forward(&transition.next_state, &self.target_q_head, &self.target_q_bias);
        let bootstrap = double_dqn_bootstrap_value(&next_online, &next_target, transition.done);
        let target = (reward + gamma * bootstrap).clamp(-10.0, 10.0);
        let td_error = (q_values[transition.action] - target).clamp(-4.0, 4.0);

        self.q_bias[transition.action] -= lr * td_error;
        for dim in 0..self.d_model {
            self.q_head[transition.action][dim] -= lr * td_error * q_repr[dim];
        }
        self.updates += 1;
        if self.updates % target_sync.max(1) == 0 {
            self.sync_target();
        }
        td_error.abs()
    }

    fn forward(
        &self,
        features: &[f32],
        q_head: &[Vec<f32>],
        q_bias: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let normalized = self.normalize(features);
        let seq_len = self.input_dim + 1;
        let mut tokens = vec![vec![0.0; self.d_model]; seq_len];
        for token_idx in 0..self.input_dim {
            for dim in 0..self.d_model {
                tokens[token_idx][dim] = self.feature_embedding[token_idx][dim]
                    + self.value_embedding[dim] * normalized[token_idx];
            }
        }
        tokens[self.input_dim].clone_from(&self.q_token);

        for layer in self.transformer_layers.iter() {
            tokens = self.forward_layer(&tokens, layer);
        }
        let q_repr = rms_norm(&tokens[self.input_dim], &self.final_rms_weight);
        let q_values = project_q_values(&q_repr, q_head, q_bias, self.actions);
        (q_repr, q_values)
    }

    fn normalize(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value - self.stats.mean[idx]) / self.stats.std[idx].max(1.0e-6))
            .collect()
    }

    fn forward_layer(&self, tokens: &[Vec<f32>], layer: &AttentionLayer) -> Vec<Vec<f32>> {
        let seq_len = tokens.len();
        let head_dim = self.d_model / self.heads;
        let kv_dim = self.kv_heads * head_dim;
        let mut q = vec![vec![0.0; self.d_model]; seq_len];
        let mut k = vec![vec![0.0; kv_dim]; seq_len];
        let mut v = vec![vec![0.0; kv_dim]; seq_len];
        for token_idx in 0..seq_len {
            let normed = rms_norm(&tokens[token_idx], &layer.rms_weight);
            q[token_idx] = mat_vec(&layer.wq, &normed);
            k[token_idx] = mat_vec(&layer.wk, &normed);
            v[token_idx] = mat_vec(&layer.wv, &normed);
        }

        let mut attended = vec![vec![0.0; self.d_model]; seq_len];
        for query_idx in 0..seq_len {
            for head_idx in 0..self.heads {
                let q_offset = head_idx * head_dim;
                let kv_head_idx = head_idx * self.kv_heads / self.heads;
                let kv_offset = kv_head_idx * head_dim;
                let scale = 1.0 / (head_dim as f32).sqrt().max(1.0);
                let mut accumulator = vec![0.0f32; head_dim];
                let mut previous_score = 0.0f32;
                let mut previous_weight = 1.0f32;
                for key_idx in 0..seq_len {
                    let mut dot = 0.0;
                    for dim in 0..head_dim {
                        dot += q[query_idx][q_offset + dim] * k[key_idx][kv_offset + dim];
                    }
                    let score = dot * scale;
                    let weight = if key_idx == 0 {
                        1.0
                    } else {
                        flash_d_recurrence_weight(score, previous_score, previous_weight)
                    };
                    for dim in 0..head_dim {
                        accumulator[dim] =
                            accumulator[dim] * (1.0 - weight) + v[key_idx][kv_offset + dim] * weight;
                    }
                    previous_score = score;
                    previous_weight = weight;
                }
                for dim in 0..head_dim {
                    attended[query_idx][q_offset + dim] = accumulator[dim];
                }
            }
        }

        let mut output = vec![vec![0.0; self.d_model]; seq_len];
        for token_idx in 0..seq_len {
            let projected = mat_vec(&layer.wo, &attended[token_idx]);
            for dim in 0..self.d_model {
                output[token_idx][dim] = tokens[token_idx][dim] + projected[dim];
            }
        }
        output
    }

    fn sync_target(&mut self) {
        self.target_q_head.clone_from(&self.q_head);
        self.target_q_bias.clone_from(&self.q_bias);
    }

    fn parameter_count(&self) -> usize {
        attention_param_count(
            self.input_dim,
            self.d_model,
            self.layers,
            self.heads,
            self.kv_heads,
            self.actions,
        )
    }

    fn to_json(&self, action_names: &[&str]) -> String {
        let mut output = String::new();
        writeln!(output, "{{").unwrap();
        writeln!(output, "  \"model_type\": \"{}\",", MODEL_TINY_Q_ATTENTION_V1).unwrap();
        writeln!(output, "  \"input_dim\": {},", self.input_dim).unwrap();
        writeln!(output, "  \"d_model\": {},", self.d_model).unwrap();
        writeln!(output, "  \"num_layers\": {},", self.layers).unwrap();
        writeln!(output, "  \"num_heads\": {},", self.heads).unwrap();
        writeln!(output, "  \"num_kv_heads\": {},", self.kv_heads).unwrap();
        writeln!(output, "  \"num_actions\": {},", self.actions).unwrap();
        write_json_array_f32(&mut output, "feature_mean", &self.stats.mean, 2, true);
        write_json_array_f32(&mut output, "feature_std", &self.stats.std, 2, true);
        write_json_matrix_f32(&mut output, "feature_embedding", &self.feature_embedding, 2, true);
        write_json_array_f32(&mut output, "value_embedding", &self.value_embedding, 2, true);
        write_json_array_f32(&mut output, "q_token", &self.q_token, 2, true);
        write_attention_layers(&mut output, &self.transformer_layers, 2, true);
        write_json_array_f32(&mut output, "final_rms_weight", &self.final_rms_weight, 2, true);
        write_json_matrix_f32(&mut output, "q_head", &self.q_head, 2, true);
        write_json_array_f32(&mut output, "q_bias", &self.q_bias, 2, true);
        write_json_array_str(&mut output, "action_names", action_names, 2, false);
        writeln!(output, "}}").unwrap();
        output
    }
}

fn ensure_parent_dir(path: &str) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn deterministic_matrix(rows: usize, cols: usize, seed: u64, fan_scale: f32) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            matrix[row][col] = deterministic_weight(seed + (row * cols + col) as u64, fan_scale);
        }
    }
    matrix
}

fn deterministic_weight(index: u64, fan_scale: f32) -> f32 {
    let mut x = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as f64 / u64::MAX as f64) as f32;
    (unit * 2.0 - 1.0) / fan_scale
}

fn mat_vec(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; matrix.len()];
    for (row_idx, row) in matrix.iter().enumerate() {
        let mut sum = 0.0;
        for (col_idx, weight) in row.iter().enumerate() {
            sum += weight * vector[col_idx];
        }
        output[row_idx] = sum;
    }
    output
}

fn rms_norm(vector: &[f32], weight: &[f32]) -> Vec<f32> {
    let mean_square =
        vector.iter().map(|value| value * value).sum::<f32>() / vector.len().max(1) as f32;
    let scale = 1.0 / (mean_square + 1.0e-6).sqrt();
    vector
        .iter()
        .enumerate()
        .map(|(idx, value)| value * scale * weight[idx])
        .collect()
}

fn project_q_values(q_repr: &[f32], q_head: &[Vec<f32>], q_bias: &[f32], actions: usize) -> Vec<f32> {
    let mut q_values = vec![0.0; actions];
    for action_idx in 0..actions {
        let mut sum = q_bias[action_idx];
        for dim in 0..q_repr.len() {
            sum += q_head[action_idx][dim] * q_repr[dim];
        }
        q_values[action_idx] = sum;
    }
    q_values
}

fn sigmoid_stable(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn flash_d_recurrence_weight(score: f32, previous_score: f32, previous_weight: f32) -> f32 {
    let log_previous_weight = previous_weight.max(1.0e-20).ln();
    sigmoid_stable(score - previous_score + log_previous_weight)
}

fn double_dqn_bootstrap_value(
    next_online_q_values: &[f32],
    next_target_q_values: &[f32],
    done: bool,
) -> f32 {
    if done || next_online_q_values.is_empty() || next_target_q_values.is_empty() {
        return 0.0;
    }
    let action_idx = argmax(next_online_q_values);
    next_target_q_values
        .get(action_idx)
        .copied()
        .unwrap_or(0.0)
        .clamp(-10.0, 10.0)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn normalize_kv_heads(kv_heads: usize, heads: usize) -> usize {
    let mut kv_heads = kv_heads.clamp(1, heads.max(1));
    if heads % kv_heads != 0 {
        kv_heads = (1..=kv_heads)
            .rev()
            .find(|candidate| heads % candidate == 0)
            .unwrap_or(1);
    }
    kv_heads
}

fn attention_param_count(
    input_dim: usize,
    d_model: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    actions: usize,
) -> usize {
    let head_dim = d_model / heads.max(1);
    let kv_dim = kv_heads.max(1) * head_dim;
    input_dim * d_model
        + d_model * 2
        + layers * (d_model + 2 * d_model * d_model + 2 * kv_dim * d_model)
        + d_model
        + actions * d_model
        + actions
}

fn write_json_array_f32(
    output: &mut String,
    name: &str,
    values: &[f32],
    indent: usize,
    comma: bool,
) {
    write!(output, "{}\"{}\": [", " ".repeat(indent), name).unwrap();
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            output.push_str(", ");
        }
        write!(output, "{:.8}", finite_value(*value)).unwrap();
    }
    if comma {
        output.push_str("],\n");
    } else {
        output.push_str("]\n");
    }
}

fn write_json_matrix_f32(
    output: &mut String,
    name: &str,
    matrix: &[Vec<f32>],
    indent: usize,
    comma: bool,
) {
    writeln!(output, "{}\"{}\": [", " ".repeat(indent), name).unwrap();
    for (row_idx, row) in matrix.iter().enumerate() {
        write!(output, "{}[", " ".repeat(indent + 2)).unwrap();
        for (col_idx, value) in row.iter().enumerate() {
            if col_idx > 0 {
                output.push_str(", ");
            }
            write!(output, "{:.8}", finite_value(*value)).unwrap();
        }
        if row_idx + 1 == matrix.len() {
            output.push_str("]\n");
        } else {
            output.push_str("],\n");
        }
    }
    if comma {
        writeln!(output, "{}],", " ".repeat(indent)).unwrap();
    } else {
        writeln!(output, "{}]", " ".repeat(indent)).unwrap();
    }
}

fn write_attention_layers(
    output: &mut String,
    layers: &[AttentionLayer],
    indent: usize,
    comma: bool,
) {
    writeln!(output, "{}\"layers\": [", " ".repeat(indent)).unwrap();
    for (idx, layer) in layers.iter().enumerate() {
        writeln!(output, "{}{{", " ".repeat(indent + 2)).unwrap();
        write_json_array_f32(output, "rms_weight", &layer.rms_weight, indent + 4, true);
        write_json_matrix_f32(output, "wq", &layer.wq, indent + 4, true);
        write_json_matrix_f32(output, "wk", &layer.wk, indent + 4, true);
        write_json_matrix_f32(output, "wv", &layer.wv, indent + 4, true);
        write_json_matrix_f32(output, "wo", &layer.wo, indent + 4, false);
        if idx + 1 == layers.len() {
            writeln!(output, "{}}}", " ".repeat(indent + 2)).unwrap();
        } else {
            writeln!(output, "{}}},", " ".repeat(indent + 2)).unwrap();
        }
    }
    if comma {
        writeln!(output, "{}],", " ".repeat(indent)).unwrap();
    } else {
        writeln!(output, "{}]", " ".repeat(indent)).unwrap();
    }
}

fn write_json_array_str(
    output: &mut String,
    name: &str,
    values: &[&str],
    indent: usize,
    comma: bool,
) {
    write!(output, "{}\"{}\": [", " ".repeat(indent), name).unwrap();
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            output.push_str(", ");
        }
        write!(output, "\"{}\"", escape_json(value)).unwrap();
    }
    if comma {
        output.push_str("],\n");
    } else {
        output.push_str("]\n");
    }
}

fn finite_value(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
