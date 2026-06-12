use serde::{Deserialize, Serialize};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

const GLOBAL_FEATURES: usize = 8;
const ENTITY_FEATURES: usize = 8;
const MAX_ZOMBIES: usize = 5;
const MAX_PARAMS: usize = 55_000;
const DEFAULT_DT: f32 = 0.20;
const DEFAULT_EPISODE_SECONDS: f32 = 60.0;
static SIM_DT: OnceLock<f32> = OnceLock::new();
static SIM_EPISODE_SECONDS: OnceLock<f32> = OnceLock::new();
const ZOMBIE_HP: f32 = 100.0;
const ZOMBIE_ATTACK_RANGE: f32 = 24.0;
const ZOMBIE_ATTACK_DAMAGE: f32 = 30.0;
const ZOMBIE_ATTACK_COOLDOWN: f32 = 0.20;
const ZOMBIE_PATH_SPEED: f32 = 44.0;
const ZOMBIE_DASH_SPEED: f32 = 96.0;
const ZOMBIE_DASH_COOLDOWN: f32 = 1.2;
const ZOMBIE_ATTACK_DOT_THRESHOLD: f32 = 0.35;
const RULE_DASH_IN_RANGE: f32 = 34.0;
const RULE_DASH_OUT_RANGE: f32 = 54.0;
const RULE_DASH_IN_CHANCE: f32 = 0.24;
const RULE_DASH_OUT_WHEN_PLAYER_ATTACKING_CHANCE: f32 = 0.46;
const PLAYER_LIGHT_DAMAGE: f32 = 25.0;
const PLAYER_HEAVY_DAMAGE: f32 = 38.0;
const PLAYER_ATTACK_DOT_THRESHOLD: f32 = 0.68;
const EPISODE_UPDATE_PASSES: usize = 2;

const ZOMBIE_ACTIONS: [&str; 8] = [
    "PATH_FRONT",
    "PATH_LEFT",
    "PATH_RIGHT",
    "PATH_BACK",
    "DASH_UP",
    "DASH_DOWN",
    "DASH_LEFT",
    "DASH_RIGHT",
];
const Z_PATH_FRONT: usize = 0;
const Z_PATH_LEFT: usize = 1;
const Z_PATH_RIGHT: usize = 2;
const Z_PATH_BACK: usize = 3;
const Z_DASH_UP: usize = 4;
const Z_DASH_DOWN: usize = 5;
const Z_DASH_LEFT: usize = 6;
const Z_DASH_RIGHT: usize = 7;

const PLAYER_ACTIONS: [&str; 16] = [
    "ROLL_UP",
    "ROLL_DOWN",
    "ROLL_FORWARD",
    "ROLL_BACK",
    "ATTACK_WHILE_MOVE_1_COMBO",
    "ATTACK_WHILE_MOVE_2_COMBOS",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "HEAL",
    "PATH_FINDING_TO_ZOMBIE_1",
    "PATH_FINDING_TO_ZOMBIE_2",
    "PATH_FINDING_TO_ZOMBIE_3",
    "PATH_FINDING_TO_ZOMBIE_4",
    "PATH_FINDING_TO_ZOMBIE_5",
];

#[derive(Clone, Copy)]
struct Config {
    d_model: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
}

impl Config {
    fn params(self, actions: usize) -> usize {
        let head_dim = self.d_model / self.heads;
        let kv_dim = self.kv_heads * head_dim;
        let feature_params = GLOBAL_FEATURES * self.d_model;
        let token_params = self.d_model * 2;
        let per_layer = self.d_model + 2 * self.d_model * self.d_model + 2 * kv_dim * self.d_model;
        let output_params = self.d_model + actions * self.d_model + actions;
        feature_params + token_params + self.layers * per_layer + output_params
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct AttentionLayer {
    rms_weight: Vec<f32>,
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Checkpoint {
    model_type: String,
    input_dim: usize,
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_actions: usize,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    feature_embedding: Vec<Vec<f32>>,
    value_embedding: Vec<f32>,
    q_token: Vec<f32>,
    layers: Vec<AttentionLayer>,
    final_rms_weight: Vec<f32>,
    q_head: Vec<Vec<f32>>,
    q_bias: Vec<f32>,
    action_names: Vec<String>,
}

#[derive(Clone)]
struct TinyEntityDecoder {
    config: Config,
    action_names: Vec<String>,
    feature_embedding: Vec<Vec<f32>>,
    value_embedding: Vec<f32>,
    q_token: Vec<f32>,
    layers: Vec<AttentionLayer>,
    final_rms_weight: Vec<f32>,
    q_head: Vec<Vec<f32>>,
    q_bias: Vec<f32>,
}

#[derive(Clone)]
struct Player {
    x: f32,
    y: f32,
    hp: f32,
    stamina: f32,
    armed: bool,
    face_x: f32,
    face_y: f32,
    attack_cooldown: f32,
    invuln: f32,
    heals: i32,
    attacking: f32,
    stuck_x: f32,
    stuck_y: f32,
}

#[derive(Clone)]
struct Zombie {
    x: f32,
    y: f32,
    hp: f32,
    face_x: f32,
    face_y: f32,
    attack_cooldown: f32,
    dash_cooldown: f32,
    stuck_x: f32,
    stuck_y: f32,
}

#[derive(Clone)]
struct State {
    global: Vec<f32>,
    entities: Vec<Vec<f32>>,
}

struct StepStats {
    player_damage_events: usize,
    player_stamina_reduced: bool,
    zombie_damage_events: usize,
    zombie_deaths: usize,
    player_damage_taken_events: usize,
    player_damage_zombie_events: usize,
    player_killed_zombies: usize,
    player_successful_dodge: bool,
    player_wasted_dodge: bool,
    player_healed_low: bool,
    player_wasted_heal: bool,
    player_wasted_attack: bool,
    zombie_idle_count: usize,
    zombie_stuck_count: usize,
    player_stuck: bool,
    mean_distance_before: f32,
    mean_distance_after: f32,
    surround: bool,
    zombie_outcomes: Vec<ZombieActionOutcome>,
}

#[derive(Clone, Copy, Default)]
struct ZombieActionOutcome {
    hit_player: bool,
    attack_in_range: bool,
    committed_attack: bool,
    stuck: bool,
}

struct EpisodeMetrics {
    player_dead: bool,
    zombies_dead: bool,
    timeout: bool,
    steps: usize,
    zombie_reward: f32,
    player_reward: f32,
    player_armed: bool,
    zombie_action_counts: Vec<usize>,
    player_action_counts: Vec<usize>,
}

struct EvalSummary {
    zombie_win_rate: f32,
    mean_ttk: f32,
    zombie_reward: f32,
    zombie_action_entropy: f32,
    player_action_entropy: f32,
    zombie_action_counts: Vec<usize>,
    player_action_counts: Vec<usize>,
}

#[derive(Clone)]
struct Transition {
    state: State,
    output_index: usize,
    output_count: usize,
    action: usize,
    reward: f32,
    next_state: State,
    done: bool,
}

impl TinyEntityDecoder {
    fn new(config: Config, action_names: &[&str], seed_offset: u64) -> Self {
        assert!(config.params(action_names.len()) <= MAX_PARAMS);
        let d_model = config.d_model;
        let d_scale = (d_model as f32).sqrt().max(1.0);
        let head_dim = d_model / config.heads;
        let kv_dim = config.kv_heads * head_dim;
        let feature_embedding = (0..GLOBAL_FEATURES)
            .map(|feature_idx| {
                (0..d_model)
                    .map(|dim| {
                        deterministic_weight(
                            seed_offset + 1_000 + (feature_idx * d_model + dim) as u64,
                            d_scale,
                        )
                    })
                    .collect()
            })
            .collect();
        let value_embedding = (0..d_model)
            .map(|dim| deterministic_weight(seed_offset + 5_000 + dim as u64, d_scale))
            .collect();
        let q_token = (0..d_model)
            .map(|dim| deterministic_weight(seed_offset + 6_000 + dim as u64, d_scale))
            .collect();
        let layers = (0..config.layers)
            .map(|layer_idx| AttentionLayer {
                rms_weight: vec![1.0; d_model],
                wq: deterministic_matrix(
                    d_model,
                    d_model,
                    seed_offset + 10_000 + layer_idx as u64 * 10_000,
                    d_scale,
                ),
                wk: deterministic_matrix(
                    kv_dim,
                    d_model,
                    seed_offset + 20_000 + layer_idx as u64 * 10_000,
                    d_scale,
                ),
                wv: deterministic_matrix(
                    kv_dim,
                    d_model,
                    seed_offset + 30_000 + layer_idx as u64 * 10_000,
                    d_scale,
                ),
                wo: deterministic_matrix(
                    d_model,
                    d_model,
                    seed_offset + 40_000 + layer_idx as u64 * 10_000,
                    d_scale,
                ),
            })
            .collect();
        let q_head = (0..action_names.len())
            .map(|action| {
                (0..d_model)
                    .map(|dim| {
                        deterministic_weight(
                            seed_offset + 70_000 + (action * d_model + dim) as u64,
                            d_scale,
                        )
                    })
                    .collect()
            })
            .collect();
        Self {
            config,
            action_names: action_names
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            feature_embedding,
            value_embedding,
            q_token,
            layers,
            final_rms_weight: vec![1.0; d_model],
            q_head,
            q_bias: vec![0.0; action_names.len()],
        }
    }

    fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        Self {
            config: Config {
                d_model: checkpoint.d_model,
                layers: checkpoint.num_layers,
                heads: checkpoint.num_heads,
                kv_heads: checkpoint.num_kv_heads,
            },
            action_names: checkpoint.action_names,
            feature_embedding: checkpoint.feature_embedding,
            value_embedding: checkpoint.value_embedding,
            q_token: checkpoint.q_token,
            layers: checkpoint.layers,
            final_rms_weight: checkpoint.final_rms_weight,
            q_head: checkpoint.q_head,
            q_bias: checkpoint.q_bias,
        }
    }

    fn params(&self) -> usize {
        self.config.params(self.action_names.len())
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            model_type: "tiny_q_attention_v1".to_string(),
            input_dim: GLOBAL_FEATURES,
            d_model: self.config.d_model,
            num_layers: self.config.layers,
            num_heads: self.config.heads,
            num_kv_heads: self.config.kv_heads,
            num_actions: self.action_names.len(),
            feature_mean: vec![0.0; GLOBAL_FEATURES],
            feature_std: vec![1.0; GLOBAL_FEATURES],
            feature_embedding: self.feature_embedding.clone(),
            value_embedding: self.value_embedding.clone(),
            q_token: self.q_token.clone(),
            layers: self.layers.clone(),
            final_rms_weight: self.final_rms_weight.clone(),
            q_head: self.q_head.clone(),
            q_bias: self.q_bias.clone(),
            action_names: self.action_names.clone(),
        }
    }

    fn action_reprs(&self, state: &State, output_count: usize) -> Vec<Vec<f32>> {
        let prefix_len = 1 + state.entities.len();
        let seq_len = prefix_len + output_count;
        let mut tokens = vec![vec![0.0; self.config.d_model]; seq_len];
        tokens[0] = self.build_global_token(&state.global);
        for (idx, entity) in state.entities.iter().enumerate() {
            tokens[1 + idx] = self.build_entity_token(entity, idx, state.entities.len());
        }
        for idx in 0..output_count {
            tokens[prefix_len + idx] = self.build_action_query_token(idx, output_count.max(1));
        }
        for layer in &self.layers {
            tokens = self.forward_causal_layer(&tokens, layer, prefix_len);
        }
        (0..output_count)
            .map(|idx| rms_norm(&tokens[prefix_len + idx], &self.final_rms_weight))
            .collect()
    }

    fn q_values_from_repr(&self, repr: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.action_names.len()];
        for action in 0..self.action_names.len() {
            output[action] = self.q_bias[action] + dot_product(&self.q_head[action], repr);
        }
        output
    }

    fn predict_actions(
        &self,
        state: &State,
        output_count: usize,
        epsilon: f32,
        rng: &mut Rng,
    ) -> Vec<usize> {
        self.action_reprs(state, output_count)
            .iter()
            .map(|repr| {
                if rng.f32() < epsilon {
                    rng.usize(0, self.action_names.len() - 1)
                } else {
                    argmax(&self.q_values_from_repr(repr))
                }
            })
            .collect()
    }

    fn learn(
        &mut self,
        state: &State,
        output_index: usize,
        output_count: usize,
        action: usize,
        reward: f32,
        next_state: &State,
        done: bool,
        gamma: f32,
        lr: f32,
    ) -> f32 {
        let reprs = self.action_reprs(state, output_count);
        if output_index >= reprs.len() || action >= self.action_names.len() {
            return 0.0;
        }
        let q_values = self.q_values_from_repr(&reprs[output_index]);
        let next_reprs = self.action_reprs(next_state, output_count);
        let bootstrap = if done || output_index >= next_reprs.len() {
            0.0
        } else {
            self.q_values_from_repr(&next_reprs[output_index])
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let target = (reward + gamma * bootstrap).clamp(-15.0, 15.0);
        let td_error = (q_values[action] - target).clamp(-5.0, 5.0);
        self.q_bias[action] -= lr * td_error;
        for dim in 0..self.config.d_model {
            self.q_head[action][dim] -= lr * td_error * reprs[output_index][dim];
        }
        td_error.abs()
    }

    fn build_global_token(&self, features: &[f32]) -> Vec<f32> {
        let mut token = vec![0.0; self.config.d_model];
        for (feature_idx, value) in features.iter().take(GLOBAL_FEATURES).enumerate() {
            let normalized = value.clamp(-4.0, 4.0);
            for dim in 0..self.config.d_model {
                token[dim] += self.feature_embedding[feature_idx][dim]
                    + self.value_embedding[dim] * normalized
                    + type_embedding(0, dim, self.config.d_model);
            }
        }
        scale_token(&mut token, GLOBAL_FEATURES);
        token
    }

    fn build_entity_token(
        &self,
        features: &[f32],
        entity_idx: usize,
        entity_count: usize,
    ) -> Vec<f32> {
        let mut token = vec![0.0; self.config.d_model];
        for (feature_idx, value) in features.iter().take(ENTITY_FEATURES).enumerate() {
            let source_idx = feature_idx % self.feature_embedding.len();
            let normalized = value.clamp(-4.0, 4.0);
            for dim in 0..self.config.d_model {
                token[dim] += self.feature_embedding[source_idx][dim] * 0.5
                    + self.value_embedding[dim] * normalized
                    + type_embedding(1, dim, self.config.d_model);
            }
        }
        let role = if entity_count <= 1 {
            0.0
        } else {
            entity_idx as f32 / (entity_count - 1) as f32
        };
        for dim in 0..self.config.d_model {
            token[dim] += self.q_token[dim] * (role - 0.5) * 0.15;
        }
        scale_token(&mut token, ENTITY_FEATURES);
        token
    }

    fn build_action_query_token(&self, output_idx: usize, output_count: usize) -> Vec<f32> {
        let mut token = self.q_token.clone();
        let role = if output_count <= 1 {
            0.0
        } else {
            output_idx as f32 / (output_count - 1) as f32
        };
        for dim in 0..self.config.d_model {
            token[dim] += type_embedding(2, dim, self.config.d_model)
                + self.value_embedding[dim] * (role - 0.5);
        }
        token
    }

    fn forward_causal_layer(
        &self,
        tokens: &[Vec<f32>],
        layer: &AttentionLayer,
        prefix_len: usize,
    ) -> Vec<Vec<f32>> {
        let seq_len = tokens.len();
        let head_dim = self.config.d_model / self.config.heads;
        let kv_dim = self.config.kv_heads * head_dim;
        let mut q = vec![vec![0.0; self.config.d_model]; seq_len];
        let mut k = vec![vec![0.0; kv_dim]; seq_len];
        let mut v = vec![vec![0.0; kv_dim]; seq_len];
        for token_idx in 0..seq_len {
            let normed = rms_norm(&tokens[token_idx], &layer.rms_weight);
            q[token_idx] = mat_vec(&layer.wq, &normed);
            k[token_idx] = mat_vec(&layer.wk, &normed);
            v[token_idx] = mat_vec(&layer.wv, &normed);
            apply_rope(&mut q[token_idx], self.config.heads, head_dim, token_idx);
            apply_rope(&mut k[token_idx], self.config.kv_heads, head_dim, token_idx);
        }

        let mut attended = vec![vec![0.0; self.config.d_model]; seq_len];
        for query_idx in 0..seq_len {
            for head_idx in 0..self.config.heads {
                let q_offset = head_idx * head_dim;
                let kv_head_idx = head_idx * self.config.kv_heads / self.config.heads;
                let kv_offset = kv_head_idx * head_dim;
                let scale = 1.0 / (head_dim as f32).sqrt().max(1.0);
                let mut accumulator = vec![0.0; head_dim];
                let mut previous_score = 0.0;
                let mut previous_weight = 1.0;
                let mut has_previous = false;
                for key_idx in 0..seq_len {
                    if !can_attend(query_idx, key_idx, prefix_len) {
                        continue;
                    }
                    let dot = dot_product(
                        &q[query_idx][q_offset..q_offset + head_dim],
                        &k[key_idx][kv_offset..kv_offset + head_dim],
                    );
                    let score = dot * scale + relative_position_bias(query_idx, key_idx);
                    let weight = if !has_previous {
                        1.0
                    } else {
                        flash_d_recurrence_weight(score, previous_score, previous_weight)
                    };
                    for dim in 0..head_dim {
                        accumulator[dim] = accumulator[dim] * (1.0 - weight)
                            + v[key_idx][kv_offset + dim] * weight;
                    }
                    previous_score = score;
                    previous_weight = weight;
                    has_previous = true;
                }
                attended[query_idx][q_offset..(head_dim + q_offset)]
                    .copy_from_slice(&accumulator[..head_dim]);
            }
        }
        tokens
            .iter()
            .enumerate()
            .map(|(token_idx, token)| {
                let projected = mat_vec(&layer.wo, &attended[token_idx]);
                (0..self.config.d_model)
                    .map(|dim| token[dim] + projected[dim])
                    .collect()
            })
            .collect()
    }
}

fn run_episode(
    zombie_policy: &mut TinyEntityDecoder,
    player_policy: &mut TinyEntityDecoder,
    rng: &mut Rng,
    epsilon_zombie: f32,
    epsilon_player: f32,
    rule_expert_zombies: bool,
    train_zombies: bool,
    train_player: bool,
    lr: f32,
    gamma: f32,
) -> EpisodeMetrics {
    let mut player = Player::new(rng);
    debug_assert!(player.armed, "self-play player must start armed");
    let zombie_count = rng.usize(1, MAX_ZOMBIES);
    let mut zombies = (0..zombie_count)
        .map(|_| Zombie::new(rng))
        .collect::<Vec<_>>();
    let mut metrics = EpisodeMetrics {
        player_dead: false,
        zombies_dead: false,
        timeout: false,
        steps: 0,
        zombie_reward: 0.0,
        player_reward: 0.0,
        player_armed: player.armed,
        zombie_action_counts: vec![0; ZOMBIE_ACTIONS.len()],
        player_action_counts: vec![0; PLAYER_ACTIONS.len()],
    };
    let mut zombie_transitions = Vec::new();
    let mut player_transitions = Vec::new();
    let max_steps = max_steps();

    for step in 0..max_steps {
        let live = live_zombie_indices(&zombies);
        if live.is_empty() {
            metrics.zombies_dead = true;
            metrics.player_reward += 10.0;
            break;
        }
        if player.hp <= 0.0 {
            metrics.player_dead = true;
            metrics.zombie_reward += 10.0;
            metrics.player_reward -= 10.0;
            break;
        }

        let state = encode_state(&player, &zombies);
        let mut zombie_actions = if rule_expert_zombies {
            rule_expert_actions(&player, &zombies, &live, rng)
        } else {
            zombie_policy.predict_actions(&state, live.len(), epsilon_zombie, rng)
        };
        sanitize_zombie_actions(&mut zombie_actions, &player, &zombies, &live);
        let player_action = sanitize_player_action(
            player_policy.predict_actions(&state, 1, epsilon_player, rng)[0],
            live.len(),
        );
        let before_zombie_distances = live
            .iter()
            .map(|idx| dist(player.x, player.y, zombies[*idx].x, zombies[*idx].y))
            .collect::<Vec<_>>();
        let before_zombie_hp = live.iter().map(|idx| zombies[*idx].hp).collect::<Vec<_>>();
        for action in &zombie_actions {
            metrics.zombie_action_counts[*action] += 1;
        }
        metrics.player_action_counts[player_action] += 1;

        let stats = apply_step(
            &mut player,
            &mut zombies,
            &live,
            &zombie_actions,
            player_action,
            rng,
        );
        let done =
            player.hp <= 0.0 || live_zombie_indices(&zombies).is_empty() || step + 1 >= max_steps;
        let (z_reward, p_reward) = rewards(
            &stats,
            player.hp <= 0.0,
            live_zombie_indices(&zombies).is_empty(),
            step + 1 >= max_steps,
            step + 1,
        );
        metrics.zombie_reward += z_reward;
        metrics.player_reward += p_reward;
        let next_state = encode_state(&player, &zombies);

        if train_zombies && !rule_expert_zombies {
            for (slot, action) in zombie_actions.iter().enumerate().take(live.len()) {
                let zombie_idx = live[slot];
                let after_distance = dist(
                    player.x,
                    player.y,
                    zombies[zombie_idx].x,
                    zombies[zombie_idx].y,
                );
                let slot_reward = zombie_slot_reward(
                    &stats,
                    slot,
                    *action,
                    zombie_actions
                        .iter()
                        .filter(|candidate| **candidate == *action)
                        .count(),
                    before_zombie_distances[slot],
                    after_distance,
                    before_zombie_hp[slot],
                    zombies[zombie_idx].hp,
                    stats.zombie_outcomes.get(slot).copied().unwrap_or_default(),
                    z_reward,
                    player.hp <= 0.0,
                    step + 1,
                );
                zombie_transitions.push(Transition {
                    state: state.clone(),
                    output_index: slot,
                    output_count: live.len(),
                    action: *action,
                    reward: slot_reward,
                    next_state: next_state.clone(),
                    done,
                });
            }
        }
        if train_player {
            player_transitions.push(Transition {
                state: state.clone(),
                output_index: 0,
                output_count: 1,
                action: player_action,
                reward: p_reward,
                next_state: next_state.clone(),
                done,
            });
        }

        metrics.steps = step + 1;
        if done {
            metrics.player_dead = player.hp <= 0.0;
            metrics.zombies_dead = live_zombie_indices(&zombies).is_empty();
            metrics.timeout =
                step + 1 >= max_steps && !metrics.player_dead && !metrics.zombies_dead;
            break;
        }
    }
    for _ in 0..EPISODE_UPDATE_PASSES {
        if train_zombies {
            for transition in &zombie_transitions {
                zombie_policy.learn(
                    &transition.state,
                    transition.output_index,
                    transition.output_count,
                    transition.action,
                    transition.reward,
                    &transition.next_state,
                    transition.done,
                    gamma,
                    lr,
                );
            }
        }
        if train_player {
            for transition in &player_transitions {
                player_policy.learn(
                    &transition.state,
                    transition.output_index,
                    transition.output_count,
                    transition.action,
                    transition.reward,
                    &transition.next_state,
                    transition.done,
                    gamma,
                    lr,
                );
            }
        }
    }
    if metrics.steps == 0 {
        metrics.steps = max_steps;
    }
    metrics
}

impl Player {
    fn new(_rng: &mut Rng) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            hp: 100.0,
            stamina: 100.0,
            armed: true,
            face_x: 1.0,
            face_y: 0.0,
            attack_cooldown: 0.0,
            invuln: 0.0,
            heals: 1,
            attacking: 0.0,
            stuck_x: 0.0,
            stuck_y: 0.0,
        }
    }
}

impl Zombie {
    fn new(rng: &mut Rng) -> Self {
        let angle = rng.range(-std::f32::consts::PI, std::f32::consts::PI);
        let distance = rng.range(56.0, 112.0);
        Self {
            x: angle.cos() * distance,
            y: angle.sin() * distance,
            hp: ZOMBIE_HP,
            face_x: -angle.cos(),
            face_y: -angle.sin(),
            attack_cooldown: rng.range(0.0, 0.3),
            dash_cooldown: rng.range(0.0, ZOMBIE_DASH_COOLDOWN),
            stuck_x: 0.0,
            stuck_y: 0.0,
        }
    }
}

fn apply_step(
    player: &mut Player,
    zombies: &mut [Zombie],
    live: &[usize],
    zombie_actions: &[usize],
    player_action: usize,
    _rng: &mut Rng,
) -> StepStats {
    let before_stamina = player.stamina;
    let mean_before = mean_distance(player, zombies, live);
    let mut stats = StepStats {
        player_damage_events: 0,
        player_stamina_reduced: false,
        zombie_damage_events: 0,
        zombie_deaths: 0,
        player_damage_taken_events: 0,
        player_damage_zombie_events: 0,
        player_killed_zombies: 0,
        player_successful_dodge: false,
        player_wasted_dodge: false,
        player_healed_low: false,
        player_wasted_heal: false,
        player_wasted_attack: false,
        zombie_idle_count: 0,
        zombie_stuck_count: 0,
        player_stuck: false,
        mean_distance_before: mean_before,
        mean_distance_after: mean_before,
        surround: false,
        zombie_outcomes: Vec::with_capacity(live.len()),
    };

    execute_player_action(player, zombies, live, player_action, &mut stats);
    for value in [
        &mut player.attack_cooldown,
        &mut player.invuln,
        &mut player.attacking,
    ] {
        *value = (*value - dt()).max(0.0);
    }
    player.stamina = (player.stamina + 10.0 * dt()).min(100.0);

    for (slot, zombie_idx) in live.iter().enumerate() {
        if zombies[*zombie_idx].hp <= 0.0 {
            continue;
        }
        let action = zombie_actions
            .get(slot)
            .copied()
            .unwrap_or_else(|| role_path_action(slot));
        let outcome = execute_zombie_action(player, &mut zombies[*zombie_idx], action, &mut stats);
        stats.zombie_outcomes.push(outcome);
    }
    for zombie in zombies.iter_mut() {
        zombie.attack_cooldown = (zombie.attack_cooldown - dt()).max(0.0);
        zombie.dash_cooldown = (zombie.dash_cooldown - dt()).max(0.0);
    }

    stats.player_stamina_reduced = player.stamina < before_stamina;
    stats.mean_distance_after = mean_distance(player, zombies, &live_zombie_indices(zombies));
    stats.surround = is_surrounded(player, zombies);
    stats
}

fn execute_player_action(
    player: &mut Player,
    zombies: &mut [Zombie],
    live: &[usize],
    action: usize,
    stats: &mut StepStats,
) {
    let nearest = nearest_zombie(player, zombies, live);
    let nearest_dir = nearest
        .map(|idx| normalized(zombies[idx].x - player.x, zombies[idx].y - player.y))
        .unwrap_or((player.face_x, player.face_y));
    match action {
        0 | 1 | 2 | 3 => {
            let dir = match action {
                0 => (0.0, -1.0),
                1 => (0.0, 1.0),
                2 => nearest_dir,
                _ => (-nearest_dir.0, -nearest_dir.1),
            };
            if player.stamina >= 20.0 {
                player.stamina -= 20.0;
                player.invuln = 0.35;
                move_actor(
                    &mut player.x,
                    &mut player.y,
                    dir,
                    120.0 * dt(),
                    &mut player.stuck_x,
                    &mut player.stuck_y,
                );
                if nearest
                    .map(|idx| dist(player.x, player.y, zombies[idx].x, zombies[idx].y) < 26.0)
                    .unwrap_or(false)
                {
                    stats.player_successful_dodge = true;
                } else {
                    stats.player_wasted_dodge = true;
                }
            } else {
                stats.player_wasted_dodge = true;
            }
        }
        4 | 5 => {
            let combo = action == 5;
            if let Some(idx) = nearest {
                let to_target = normalized(zombies[idx].x - player.x, zombies[idx].y - player.y);
                player.face_x = to_target.0;
                player.face_y = to_target.1;
                let attack_range = if combo { 34.0 } else { 28.0 };
                let attack_distance = dist(player.x, player.y, zombies[idx].x, zombies[idx].y);
                if attack_distance > attack_range {
                    move_actor(
                        &mut player.x,
                        &mut player.y,
                        to_target,
                        68.0 * dt(),
                        &mut player.stuck_x,
                        &mut player.stuck_y,
                    );
                }
                if dist(player.x, player.y, zombies[idx].x, zombies[idx].y) > attack_range {
                    return;
                }
            } else {
                stats.player_wasted_attack = true;
                return;
            }

            if !player.armed {
                stats.player_wasted_attack = true;
            } else if player.attack_cooldown <= 0.0
                && player.stamina >= if combo { 18.0 } else { 10.0 }
            {
                player.stamina -= if combo { 18.0 } else { 10.0 };
                player.attack_cooldown = if combo { 0.55 } else { 0.30 };
                player.attacking = 0.25;
                if let Some(idx) = nearest {
                    let attack_range = if combo { 34.0 } else { 28.0 };
                    let to_target =
                        normalized(zombies[idx].x - player.x, zombies[idx].y - player.y);
                    player.face_x = to_target.0;
                    player.face_y = to_target.1;
                    let facing_dot = player.face_x * to_target.0 + player.face_y * to_target.1;
                    if dist(player.x, player.y, zombies[idx].x, zombies[idx].y) <= attack_range
                        && facing_dot >= PLAYER_ATTACK_DOT_THRESHOLD
                    {
                        let damage = if combo {
                            PLAYER_HEAVY_DAMAGE
                        } else {
                            PLAYER_LIGHT_DAMAGE
                        };
                        zombies[idx].hp -= damage;
                        stats.player_damage_zombie_events += 1;
                        stats.zombie_damage_events += 1;
                        if zombies[idx].hp <= 0.0 {
                            stats.player_killed_zombies += 1;
                            stats.zombie_deaths += 1;
                        }
                    } else {
                        stats.player_wasted_attack = true;
                    }
                } else {
                    stats.player_wasted_attack = true;
                }
            } else {
                stats.player_wasted_attack = true;
            }
        }
        6 | 7 | 8 | 9 => {
            let dir = match action {
                6 => (0.0, -1.0),
                7 => (0.0, 1.0),
                8 => (1.0, 0.0),
                _ => (-1.0, 0.0),
            };
            player.face_x = dir.0;
            player.face_y = dir.1;
            let before = (player.x, player.y);
            move_actor(
                &mut player.x,
                &mut player.y,
                dir,
                55.0 * dt(),
                &mut player.stuck_x,
                &mut player.stuck_y,
            );
            stats.player_stuck = dist(before.0, before.1, player.x, player.y) < 0.1;
        }
        10 => {
            if player.heals > 0 && player.hp < 45.0 {
                player.hp = (player.hp + 35.0).min(100.0);
                player.heals -= 1;
                stats.player_healed_low = true;
            } else {
                stats.player_wasted_heal = true;
            }
        }
        11..=15 => {
            let target_idx = action - 11;
            if target_idx < live.len() {
                let zombie = &zombies[live[target_idx]];
                let dir = normalized(zombie.x - player.x, zombie.y - player.y);
                player.face_x = dir.0;
                player.face_y = dir.1;
                move_actor(
                    &mut player.x,
                    &mut player.y,
                    dir,
                    62.0 * dt(),
                    &mut player.stuck_x,
                    &mut player.stuck_y,
                );
            }
        }
        _ => {}
    }
}

fn execute_zombie_action(
    player: &mut Player,
    zombie: &mut Zombie,
    action: usize,
    stats: &mut StepStats,
) -> ZombieActionOutcome {
    let before = (zombie.x, zombie.y);
    let mut outcome = ZombieActionOutcome::default();
    match action {
        Z_PATH_FRONT | Z_PATH_LEFT | Z_PATH_RIGHT | Z_PATH_BACK => {
            let target = zombie_slot_target(player, action);
            let dir = normalized(target.0 - zombie.x, target.1 - zombie.y);
            zombie.face_x = dir.0;
            zombie.face_y = dir.1;
            move_actor(
                &mut zombie.x,
                &mut zombie.y,
                dir,
                ZOMBIE_PATH_SPEED * dt(),
                &mut zombie.stuck_x,
                &mut zombie.stuck_y,
            );
        }
        Z_DASH_UP | Z_DASH_DOWN | Z_DASH_LEFT | Z_DASH_RIGHT => {
            if zombie.dash_cooldown <= 0.0 {
                zombie.dash_cooldown = ZOMBIE_DASH_COOLDOWN;
                let dir = match action {
                    Z_DASH_UP => (0.0, -1.0),
                    Z_DASH_DOWN => (0.0, 1.0),
                    Z_DASH_LEFT => (-1.0, 0.0),
                    _ => (1.0, 0.0),
                };
                zombie.face_x = dir.0;
                zombie.face_y = dir.1;
                move_actor(
                    &mut zombie.x,
                    &mut zombie.y,
                    dir,
                    ZOMBIE_DASH_SPEED * dt(),
                    &mut zombie.stuck_x,
                    &mut zombie.stuck_y,
                );
            } else {
                stats.zombie_idle_count += 1;
            }
        }
        _ => {}
    }
    try_zombie_auto_attack(player, zombie, stats, &mut outcome);
    if dist(before.0, before.1, zombie.x, zombie.y) < 0.1 {
        stats.zombie_stuck_count += 1;
        outcome.stuck = true;
    }
    outcome
}

fn try_zombie_auto_attack(
    player: &mut Player,
    zombie: &mut Zombie,
    stats: &mut StepStats,
    outcome: &mut ZombieActionOutcome,
) {
    if zombie.attack_cooldown > 0.0 {
        return;
    }
    let attack_distance = dist(player.x, player.y, zombie.x, zombie.y);
    if attack_distance > ZOMBIE_ATTACK_RANGE {
        return;
    }

    let to_player = normalized(player.x - zombie.x, player.y - zombie.y);
    zombie.face_x = to_player.0;
    zombie.face_y = to_player.1;
    let facing_dot = zombie.face_x * to_player.0 + zombie.face_y * to_player.1;
    if facing_dot < ZOMBIE_ATTACK_DOT_THRESHOLD {
        return;
    }

    zombie.attack_cooldown = ZOMBIE_ATTACK_COOLDOWN;
    outcome.attack_in_range = true;
    outcome.committed_attack = true;
    if player.invuln <= 0.0 {
        player.hp -= ZOMBIE_ATTACK_DAMAGE;
        stats.player_damage_events += 1;
        stats.player_damage_taken_events += 1;
        outcome.hit_player = true;
    }
}

fn rewards(
    stats: &StepStats,
    player_dead: bool,
    zombies_dead: bool,
    timeout: bool,
    steps_elapsed: usize,
) -> (f32, f32) {
    let mut zombie_reward = 0.0;
    zombie_reward += stats.player_damage_events as f32 * 2.4;
    if stats.player_stamina_reduced {
        zombie_reward += 0.2;
    }
    if stats.mean_distance_after < stats.mean_distance_before {
        zombie_reward += 0.18;
    }
    if stats.surround {
        zombie_reward += 0.55;
    }
    zombie_reward -= stats.zombie_damage_events as f32 * 0.8;
    zombie_reward -= stats.zombie_deaths as f32 * 4.0;
    zombie_reward -= stats.zombie_idle_count as f32 * 0.12;
    zombie_reward -= stats.zombie_stuck_count as f32 * 0.18;
    if player_dead {
        let remaining = max_steps().saturating_sub(steps_elapsed) as f32;
        zombie_reward += 10.0 + remaining * 0.035;
    }

    let mut player_reward = 0.0;
    if zombies_dead {
        player_reward += 12.0;
    } else if timeout {
        player_reward += 1.0;
    }
    player_reward += stats.player_damage_zombie_events as f32 * 1.0;
    player_reward += stats.player_killed_zombies as f32 * 3.0;
    if stats.player_successful_dodge {
        player_reward += 0.3;
    }
    if stats.player_wasted_dodge {
        player_reward -= 0.25;
    }
    if stats.player_healed_low {
        player_reward += 0.5;
    }
    if stats.player_wasted_heal {
        player_reward -= 0.35;
    }
    if player_dead {
        player_reward -= 10.0;
    }
    player_reward -= stats.player_damage_taken_events as f32 * 1.0;
    if stats.player_stuck {
        player_reward -= 0.1;
    }
    if stats.player_wasted_attack {
        player_reward -= 0.05;
    }
    (
        zombie_reward.clamp(-20.0, 25.0),
        player_reward.clamp(-15.0, 15.0),
    )
}

fn zombie_slot_reward(
    stats: &StepStats,
    slot: usize,
    action: usize,
    same_action_count: usize,
    before_distance: f32,
    after_distance: f32,
    before_hp: f32,
    after_hp: f32,
    outcome: ZombieActionOutcome,
    global_reward: f32,
    player_dead: bool,
    steps_elapsed: usize,
) -> f32 {
    let mut reward = global_reward * 0.12;
    let distance_delta = before_distance - after_distance;
    reward += distance_delta.clamp(-12.0, 12.0) * 0.035;
    let expected_role_action = role_path_action(slot);

    if outcome.attack_in_range {
        reward += 0.75;
    }
    if outcome.committed_attack {
        reward += 0.20;
    }
    if outcome.hit_player {
        reward += 3.4;
    }
    if after_distance <= ZOMBIE_ATTACK_RANGE {
        reward += 0.25;
    }

    if action == expected_role_action {
        reward += 0.85;
    } else if matches!(
        action,
        Z_PATH_FRONT | Z_PATH_LEFT | Z_PATH_RIGHT | Z_PATH_BACK
    ) {
        reward -= 0.35;
    }

    if matches!(action, Z_PATH_LEFT | Z_PATH_RIGHT | Z_PATH_BACK) && after_distance <= 54.0 {
        reward += 0.34;
    }
    if matches!(action, Z_DASH_UP | Z_DASH_DOWN | Z_DASH_LEFT | Z_DASH_RIGHT) {
        if after_distance < before_distance {
            reward += 0.30;
        } else {
            reward -= 0.20;
        }
    }
    if same_action_count <= 1 {
        reward += 0.45;
    } else {
        reward -= 0.20 * (same_action_count.saturating_sub(1) as f32);
    }
    if stats.surround {
        reward += 0.20;
    }
    if outcome.stuck {
        reward -= 0.45;
    }
    if before_hp > after_hp {
        reward -= 1.0;
    }
    if before_hp > 0.0 && after_hp <= 0.0 {
        reward -= 5.0;
    }
    if player_dead {
        let remaining = max_steps().saturating_sub(steps_elapsed) as f32;
        reward += 3.0 + remaining * 0.012;
    }
    reward.clamp(-8.0, 8.0)
}

fn encode_state(player: &Player, zombies: &[Zombie]) -> State {
    let live = live_zombie_indices(zombies);
    let global = vec![
        player.x / 128.0,
        player.y / 128.0,
        player.stuck_x,
        player.stuck_y,
        player.stamina / 100.0,
        player.attacking.clamp(0.0, 1.0),
        player.hp / 100.0,
        live.len() as f32 / MAX_ZOMBIES as f32,
    ];
    let mut entities = Vec::new();
    for idx in live {
        let zombie = &zombies[idx];
        entities.push(vec![
            zombie.x / 128.0,
            zombie.y / 128.0,
            zombie.stuck_x,
            zombie.stuck_y,
            zombie.hp / ZOMBIE_HP,
            zombie.attack_cooldown / 2.0,
            zombie.dash_cooldown / 4.0,
            dist(player.x, player.y, zombie.x, zombie.y) / 220.0,
        ]);
    }
    State { global, entities }
}

fn zombie_slot_target(player: &Player, action: usize) -> (f32, f32) {
    let right = (-player.face_y, player.face_x);
    match action {
        Z_PATH_FRONT => (
            player.x + player.face_x * 18.0,
            player.y + player.face_y * 18.0,
        ),
        Z_PATH_LEFT => (player.x - right.0 * 24.0, player.y - right.1 * 24.0),
        Z_PATH_RIGHT => (player.x + right.0 * 24.0, player.y + right.1 * 24.0),
        Z_PATH_BACK => (
            player.x - player.face_x * 22.0,
            player.y - player.face_y * 22.0,
        ),
        _ => (player.x, player.y),
    }
}

fn sanitize_zombie_actions(
    actions: &mut [usize],
    _player: &Player,
    zombies: &[Zombie],
    live: &[usize],
) {
    for (slot, action) in actions.iter_mut().enumerate() {
        if slot >= live.len() {
            continue;
        }
        let zombie = &zombies[live[slot]];
        let path_action = role_path_action(slot);
        if matches!(
            *action,
            Z_DASH_UP | Z_DASH_DOWN | Z_DASH_LEFT | Z_DASH_RIGHT
        ) && zombie.dash_cooldown > 0.0
        {
            *action = path_action;
        }
    }
}

fn rule_expert_actions(
    player: &Player,
    zombies: &[Zombie],
    live: &[usize],
    rng: &mut Rng,
) -> Vec<usize> {
    if live.is_empty() {
        return Vec::new();
    }
    let nearest_distance = live
        .iter()
        .map(|idx| dist(player.x, player.y, zombies[*idx].x, zombies[*idx].y))
        .fold(f32::INFINITY, f32::min);
    let global_action = if nearest_distance <= 30.0 || live.len() >= 3 {
        choose_open_path_action(player, zombies, live)
    } else {
        Z_PATH_FRONT
    };
    live.iter()
        .enumerate()
        .map(|(slot, idx)| {
            let zombie = &zombies[*idx];
            let distance_to_player = dist(player.x, player.y, zombie.x, zombie.y);
            let fallback = if nearest_distance <= 30.0 {
                role_path_action(slot)
            } else {
                global_action
            };
            choose_rule_zombie_action(player, zombie, distance_to_player, fallback, rng)
        })
        .collect()
}

fn choose_rule_zombie_action(
    player: &Player,
    zombie: &Zombie,
    distance_to_player: f32,
    fallback: usize,
    rng: &mut Rng,
) -> usize {
    if zombie.dash_cooldown > 0.0 {
        return fallback;
    }

    let to_player = (player.x - zombie.x, player.y - zombie.y);
    if player.attacking > 0.0
        && distance_to_player <= RULE_DASH_OUT_RANGE
        && rng.f32() < RULE_DASH_OUT_WHEN_PLAYER_ATTACKING_CHANCE
    {
        return dash_action_from_direction((-to_player.0, -to_player.1));
    }

    if distance_to_player <= RULE_DASH_IN_RANGE && rng.f32() < RULE_DASH_IN_CHANCE {
        return dash_action_from_direction(to_player);
    }

    fallback
}

fn dash_action_from_direction(direction: (f32, f32)) -> usize {
    if direction.0.abs() >= direction.1.abs() {
        if direction.0 >= 0.0 {
            Z_DASH_RIGHT
        } else {
            Z_DASH_LEFT
        }
    } else if direction.1 >= 0.0 {
        Z_DASH_DOWN
    } else {
        Z_DASH_UP
    }
}

fn choose_open_path_action(player: &Player, zombies: &[Zombie], live: &[usize]) -> usize {
    let mut left_pressure = 0.0;
    let mut right_pressure = 0.0;
    let mut back_pressure = 0.0;
    let right = (-player.face_y, player.face_x);
    for idx in live {
        let zombie = &zombies[*idx];
        let dx = zombie.x - player.x;
        let dy = zombie.y - player.y;
        let distance = (dx * dx + dy * dy).sqrt().max(1.0);
        let side_dot = (dx * right.0 + dy * right.1) / distance;
        let facing_dot = (dx * player.face_x + dy * player.face_y) / distance;
        let pressure = 1.0 / (1.0 + distance / 48.0);
        if facing_dot < -0.25 {
            back_pressure += pressure;
        }
        if side_dot < -0.12 {
            left_pressure += pressure;
        } else if side_dot > 0.12 {
            right_pressure += pressure;
        }
    }
    if back_pressure < left_pressure.min(right_pressure) - 0.05 {
        return Z_PATH_BACK;
    }
    if left_pressure < right_pressure - 0.05 {
        return Z_PATH_LEFT;
    }
    if right_pressure < left_pressure - 0.05 {
        return Z_PATH_RIGHT;
    }
    if left_pressure <= right_pressure {
        Z_PATH_LEFT
    } else {
        Z_PATH_RIGHT
    }
}

fn sanitize_player_action(action: usize, live_count: usize) -> usize {
    if (11..=15).contains(&action) && action - 11 >= live_count {
        return 11 + live_count.saturating_sub(1);
    }
    action
}

fn role_path_action(slot: usize) -> usize {
    match slot % 4 {
        0 => Z_PATH_FRONT,
        1 => Z_PATH_LEFT,
        2 => Z_PATH_RIGHT,
        _ => Z_PATH_BACK,
    }
}

fn write_checkpoint(path: &Path, policy: &TinyEntityDecoder) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create checkpoint directory");
    }
    let json = serde_json::to_string_pretty(&policy.checkpoint()).expect("serialize checkpoint");
    fs::write(path, json).expect("write checkpoint");
}

fn read_checkpoint(path: &Path) -> Option<TinyEntityDecoder> {
    let text = fs::read_to_string(path).ok()?;
    let checkpoint = serde_json::from_str::<Checkpoint>(&text).ok()?;
    Some(TinyEntityDecoder::from_checkpoint(checkpoint))
}

fn write_report(path: &Path, lines: &[String]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create report directory");
    }
    fs::write(path, lines.join("\n")).expect("write report");
}

fn read_best_from_report(path: &Path) -> Option<(f32, usize)> {
    let text = fs::read_to_string(path).ok()?;
    let mut best: Option<(f32, usize)> = None;
    for line in text.lines().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.len() < 13 {
            continue;
        }
        let score = columns.get(9)?.parse::<f32>().ok()?;
        let best_episode = columns.get(12)?.parse::<usize>().ok()?;
        if best
            .map(|(best_score, _)| score > best_score)
            .unwrap_or(true)
        {
            best = Some((score, best_episode));
        }
    }
    best
}

fn write_replay_sample(path: &Path, episode: usize, metrics: &EpisodeMetrics) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create replay directory");
    }
    let text = format!(
        "episode,steps,player_armed,player_dead,zombies_dead,timeout,zombie_reward,player_reward\n{},{},{},{},{},{},{:.4},{:.4}\n",
        episode,
        metrics.steps,
        metrics.player_armed,
        metrics.player_dead,
        metrics.zombies_dead,
        metrics.timeout,
        metrics.zombie_reward,
        metrics.player_reward
    );
    fs::write(path, text).expect("write replay sample");
}

fn evaluate(
    zombie_policy: &mut TinyEntityDecoder,
    player_policy: &mut TinyEntityDecoder,
    seed: u64,
    episodes: usize,
    rule_expert_zombies: bool,
) -> EvalSummary {
    let mut rng = Rng::new(seed);
    let mut zombie_wins = 0usize;
    let mut mean_steps = 0.0;
    let mut mean_zombie_reward = 0.0;
    let mut zombie_action_counts = vec![0usize; ZOMBIE_ACTIONS.len()];
    let mut player_action_counts = vec![0usize; PLAYER_ACTIONS.len()];
    for _ in 0..episodes {
        let metrics = run_episode(
            zombie_policy,
            player_policy,
            &mut rng,
            0.0,
            0.0,
            rule_expert_zombies,
            false,
            false,
            0.0,
            0.96,
        );
        zombie_wins += usize::from(metrics.player_dead);
        mean_steps += metrics.steps as f32;
        mean_zombie_reward += metrics.zombie_reward;
        for (idx, count) in metrics.zombie_action_counts.iter().enumerate() {
            zombie_action_counts[idx] += *count;
        }
        for (idx, count) in metrics.player_action_counts.iter().enumerate() {
            player_action_counts[idx] += *count;
        }
    }
    EvalSummary {
        zombie_win_rate: zombie_wins as f32 / episodes.max(1) as f32,
        mean_ttk: mean_steps / episodes.max(1) as f32 * dt(),
        zombie_reward: mean_zombie_reward / episodes.max(1) as f32,
        zombie_action_entropy: normalized_entropy(&zombie_action_counts),
        player_action_entropy: normalized_entropy(&player_action_counts),
        zombie_action_counts,
        player_action_counts,
    }
}

fn normalized_entropy(counts: &[usize]) -> f32 {
    let total = counts.iter().sum::<usize>() as f32;
    if total <= 0.0 || counts.len() <= 1 {
        return 0.0;
    }
    let entropy = counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f32 / total;
            -p * p.ln()
        })
        .sum::<f32>();
    entropy / (counts.len() as f32).ln()
}

fn eval_score(summary: &EvalSummary) -> f32 {
    summary.zombie_win_rate * 100.0 - summary.mean_ttk
        + summary.zombie_action_entropy * 8.0
        + summary.zombie_reward * 0.05
}

fn action_histogram(names: &[&str], counts: &[usize]) -> String {
    let total = counts.iter().sum::<usize>().max(1) as f32;
    names
        .iter()
        .zip(counts.iter())
        .map(|(name, count)| format!("{}:{:.2}", name, *count as f32 / total))
        .collect::<Vec<_>>()
        .join(" ")
}

fn capacity_sweep_configs() -> Vec<(usize, Config)> {
    vec![
        (
            1_000,
            Config {
                d_model: 16,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            2_000,
            Config {
                d_model: 22,
                layers: 1,
                heads: 2,
                kv_heads: 1,
            },
        ),
        (
            4_000,
            Config {
                d_model: 36,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            8_000,
            Config {
                d_model: 52,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            10_000,
            Config {
                d_model: 60,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            20_000,
            Config {
                d_model: 84,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            30_000,
            Config {
                d_model: 104,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            40_000,
            Config {
                d_model: 124,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
        (
            50_000,
            Config {
                d_model: 136,
                layers: 1,
                heads: 4,
                kv_heads: 1,
            },
        ),
    ]
}

fn benchmark_zombie_latency_us(policy: &TinyEntityDecoder, iterations: usize, seed: u64) -> f32 {
    let mut rng = Rng::new(seed);
    let player = Player::new(&mut rng);
    let zombies = (0..MAX_ZOMBIES)
        .map(|_| Zombie::new(&mut rng))
        .collect::<Vec<_>>();
    let state = encode_state(&player, &zombies);
    let warmup = 100usize;
    let mut checksum = 0usize;
    for _ in 0..warmup {
        for action in policy.predict_actions(&state, MAX_ZOMBIES, 0.0, &mut rng) {
            checksum = checksum.wrapping_add(action);
        }
    }
    let started = Instant::now();
    for _ in 0..iterations {
        for action in policy.predict_actions(&state, MAX_ZOMBIES, 0.0, &mut rng) {
            checksum = checksum.wrapping_add(action);
        }
    }
    black_box(checksum);
    started.elapsed().as_secs_f32() * 1_000_000.0 / iterations.max(1) as f32
}

fn run_capacity_sweep(
    episodes: usize,
    seed: u64,
    lr: f32,
    gamma: f32,
    epsilon_start: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    output_dir: &str,
) {
    let eval_episodes = parse_arg("eval-episodes", 128usize);
    let latency_iters = parse_arg("latency-iters", 2_000usize);
    let sweep_max_target = parse_arg("sweep-max-target", usize::MAX);
    let sweep_report_path = std::env::args()
        .find_map(|arg| arg.strip_prefix("--sweep-report=").map(ToOwned::to_owned))
        .unwrap_or_else(|| "docs/ai/entity_decoder_capacity_sweep_1k_50k.csv".to_string());
    let mut rows = vec![
        "target_budget,zombie_params,player_params,d_model,layers,heads,kv_heads,episodes,eval_episodes,zombie_win_rate,mean_ttk,zombie_reward,zombie_action_entropy,player_action_entropy,eval_score,latency_us,action_histogram".to_string(),
    ];
    let mut best_score = f32::NEG_INFINITY;
    let mut best_target = 0usize;
    let mut best_zombie_policy: Option<TinyEntityDecoder> = None;
    let mut best_player_policy: Option<TinyEntityDecoder> = None;

    for (target_budget, config) in capacity_sweep_configs()
        .into_iter()
        .filter(|(target_budget, _)| *target_budget <= sweep_max_target)
    {
        let zombie_params = config.params(ZOMBIE_ACTIONS.len());
        let player_params = config.params(PLAYER_ACTIONS.len());
        println!(
            "sweep target={} zombie_params={} player_params={} d={} layers={} heads={} kv_heads={}",
            target_budget,
            zombie_params,
            player_params,
            config.d_model,
            config.layers,
            config.heads,
            config.kv_heads
        );

        let mut zombie_policy = TinyEntityDecoder::new(config, &ZOMBIE_ACTIONS, 0);
        let mut player_policy = TinyEntityDecoder::new(config, &PLAYER_ACTIONS, 900_000);
        let mut rng = Rng::new(seed);
        let mut epsilon_zombie = epsilon_start;
        let mut epsilon_player = epsilon_start;

        for _episode in 1..=episodes {
            run_episode(
                &mut zombie_policy,
                &mut player_policy,
                &mut rng,
                epsilon_zombie,
                epsilon_player,
                false,
                true,
                true,
                lr,
                gamma,
            );
            epsilon_zombie = (epsilon_zombie * epsilon_decay).max(epsilon_min);
            epsilon_player = (epsilon_player * epsilon_decay).max(epsilon_min);
        }

        let summary = evaluate(
            &mut zombie_policy,
            &mut player_policy,
            seed + 20_000,
            eval_episodes,
            false,
        );
        let latency_us =
            benchmark_zombie_latency_us(&zombie_policy, latency_iters, seed + target_budget as u64);
        let score = eval_score(&summary);
        let histogram = action_histogram(&ZOMBIE_ACTIONS, &summary.zombie_action_counts);
        println!(
            "sweep result target={} win={:.3} ttk={:.3}s entropy={:.3} score={:.3} latency={:.3}us actions={}",
            target_budget,
            summary.zombie_win_rate,
            summary.mean_ttk,
            summary.zombie_action_entropy,
            score,
            latency_us,
            histogram
        );
        rows.push(format!(
            "{},{},{},{},{},{},{},{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},\"{}\"",
            target_budget,
            zombie_params,
            player_params,
            config.d_model,
            config.layers,
            config.heads,
            config.kv_heads,
            episodes,
            eval_episodes,
            summary.zombie_win_rate,
            summary.mean_ttk,
            summary.zombie_reward,
            summary.zombie_action_entropy,
            summary.player_action_entropy,
            score,
            latency_us,
            histogram
        ));
        write_report(Path::new(&sweep_report_path), &rows);

        if latency_us >= 1_000.0 {
            println!(
                "sweep stopped: target={} reached millisecond latency ({:.3}us)",
                target_budget, latency_us
            );
            break;
        }
        if score > best_score {
            best_score = score;
            best_target = target_budget;
            best_zombie_policy = Some(zombie_policy.clone());
            best_player_policy = Some(player_policy.clone());
            write_checkpoint(
                Path::new(&format!("{}/zombie_policy_sweep_best.json", output_dir)),
                &zombie_policy,
            );
            write_checkpoint(
                Path::new(&format!("{}/player_policy_sweep_best.json", output_dir)),
                &player_policy,
            );
        }
    }

    if let (Some(zombie_policy), Some(player_policy)) = (best_zombie_policy, best_player_policy) {
        write_checkpoint(Path::new("checkpoints/zombie_policy.json"), &zombie_policy);
        write_checkpoint(
            Path::new("checkpoints/zombie_policy_attention_release.json"),
            &zombie_policy,
        );
        write_checkpoint(Path::new("checkpoints/player_policy.json"), &player_policy);
        write_checkpoint(
            Path::new(&format!(
                "{}/zombie_policy_sweep_best_target_{}.json",
                output_dir, best_target
            )),
            &zombie_policy,
        );
        write_checkpoint(
            Path::new(&format!(
                "{}/player_policy_sweep_best_target_{}.json",
                output_dir, best_target
            )),
            &player_policy,
        );
        write_checkpoint(
            Path::new(&format!("{}/zombie_policy_final.json", output_dir)),
            &zombie_policy,
        );
        write_checkpoint(
            Path::new(&format!("{}/player_policy_final.json", output_dir)),
            &player_policy,
        );
        println!(
            "sweep best target={} score={:.3}; installed checkpoints/zombie_policy.json",
            best_target, best_score
        );
    }
}

fn main() {
    let episodes = parse_arg("episodes", 10_000usize);
    let seed = parse_arg("seed", 159u64);
    let lr = parse_arg("lr", 0.004f32);
    let gamma = parse_arg("gamma", 0.96f32);
    let sim_dt = parse_arg("dt", DEFAULT_DT).clamp(0.01, 0.5);
    let sim_episode_seconds =
        parse_arg("episode-seconds", DEFAULT_EPISODE_SECONDS).clamp(1.0, 600.0);
    configure_sim_timing(sim_dt, sim_episode_seconds);
    let epsilon_start = parse_arg("epsilon", 1.0f32);
    let epsilon_min = parse_arg("epsilon-min", 0.10f32);
    let epsilon_decay = parse_arg("epsilon-decay", 0.9975f32);
    let output_dir = std::env::args()
        .find_map(|arg| arg.strip_prefix("--output-dir=").map(ToOwned::to_owned))
        .unwrap_or_else(|| "checkpoints/entity_decoder_selfplay".to_string());
    let report_path = std::env::args()
        .find_map(|arg| arg.strip_prefix("--report=").map(ToOwned::to_owned))
        .unwrap_or_else(|| "docs/ai/entity_decoder_selfplay_report.csv".to_string());
    let resume = parse_bool_arg("resume", false);
    let compare_rule_expert = parse_bool_arg("compare-rule-expert", false);
    let sweep_capacity = parse_bool_arg("sweep-capacity", false);
    let episode_offset = parse_arg("episode-offset", 0usize);
    let mut epsilon_zombie =
        decayed_epsilon(epsilon_start, epsilon_decay, epsilon_min, episode_offset);
    let mut epsilon_player =
        decayed_epsilon(epsilon_start, epsilon_decay, epsilon_min, episode_offset);

    let config = Config {
        d_model: 22,
        layers: 1,
        heads: 2,
        kv_heads: 1,
    };
    let zombie_resume_path = format!("{}/zombie_policy_final.json", output_dir);
    let player_resume_path = format!("{}/player_policy_final.json", output_dir);
    let mut zombie_policy = if resume {
        read_checkpoint(Path::new(&zombie_resume_path))
            .unwrap_or_else(|| TinyEntityDecoder::new(config, &ZOMBIE_ACTIONS, 0))
    } else {
        TinyEntityDecoder::new(config, &ZOMBIE_ACTIONS, 0)
    };
    let mut player_policy = if resume {
        read_checkpoint(Path::new(&player_resume_path))
            .unwrap_or_else(|| TinyEntityDecoder::new(config, &PLAYER_ACTIONS, 900_000))
    } else {
        TinyEntityDecoder::new(config, &PLAYER_ACTIONS, 900_000)
    };
    let loadout_probe = Player::new(&mut Rng::new(seed ^ 0xA11CE));
    assert!(
        loadout_probe.armed,
        "self-play player must start armed for duelist training"
    );

    if sweep_capacity {
        println!(
            "capacity sweep active: episodes_per_config={} player_armed=true latency_stop_us=1000",
            episodes
        );
        run_capacity_sweep(
            episodes,
            seed,
            lr,
            gamma,
            epsilon_start,
            epsilon_min,
            epsilon_decay,
            &output_dir,
        );
        return;
    }

    let mut best_zombie_policy = if resume {
        read_checkpoint(Path::new(&format!(
            "{}/zombie_policy_best.json",
            output_dir
        )))
        .unwrap_or_else(|| zombie_policy.clone())
    } else {
        zombie_policy.clone()
    };
    let mut best_player_policy = if resume {
        read_checkpoint(Path::new(&format!(
            "{}/player_policy_best.json",
            output_dir
        )))
        .unwrap_or_else(|| player_policy.clone())
    } else {
        player_policy.clone()
    };
    let (mut best_score, mut best_episode) = if resume {
        read_best_from_report(Path::new(&report_path)).unwrap_or((f32::NEG_INFINITY, 0usize))
    } else {
        (f32::NEG_INFINITY, 0usize)
    };
    let mut rng = Rng::new(seed);
    let mut report = if resume && Path::new(&report_path).exists() {
        fs::read_to_string(&report_path)
            .unwrap_or_default()
            .lines()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    } else {
        vec![
            "episode,phase,epsilon_zombie,epsilon_player,zombie_win_rate,mean_ttk,zombie_reward,zombie_action_entropy,player_action_entropy,eval_score,zombie_params,player_params,best_episode,dt,max_steps".to_string(),
        ]
    };
    if report.is_empty() {
        report.push("episode,phase,epsilon_zombie,epsilon_player,zombie_win_rate,mean_ttk,zombie_reward,zombie_action_entropy,player_action_entropy,eval_score,zombie_params,player_params,best_episode,dt,max_steps".to_string());
    }

    if compare_rule_expert {
        let eval_episodes = parse_arg("eval-episodes", 256usize);
        let learned = evaluate(
            &mut zombie_policy,
            &mut player_policy,
            seed + 10_000,
            eval_episodes,
            false,
        );
        let rule = evaluate(
            &mut zombie_policy,
            &mut player_policy,
            seed + 10_000,
            eval_episodes,
            true,
        );
        println!(
            "compare eval_episodes={} dt={:.3} max_steps={} player_armed={} zombie_params={} player_params={}",
            eval_episodes,
            dt(),
            max_steps(),
            loadout_probe.armed,
            zombie_policy.params(),
            player_policy.params()
        );
        println!(
            "learned_attention zombie_win={:.3} mean_ttk={:.3}s z_reward={:.3} z_entropy={:.3} score={:.3}",
            learned.zombie_win_rate,
            learned.mean_ttk,
            learned.zombie_reward,
            learned.zombie_action_entropy,
            eval_score(&learned)
        );
        println!(
            "learned_actions {}",
            action_histogram(&ZOMBIE_ACTIONS, &learned.zombie_action_counts)
        );
        println!(
            "rule_expert zombie_win={:.3} mean_ttk={:.3}s z_reward={:.3} z_entropy={:.3} score={:.3}",
            rule.zombie_win_rate,
            rule.mean_ttk,
            rule.zombie_reward,
            rule.zombie_action_entropy,
            eval_score(&rule)
        );
        println!(
            "rule_actions {}",
            action_histogram(&ZOMBIE_ACTIONS, &rule.zombie_action_counts)
        );
        return;
    }

    println!(
        "entity-decoder simultaneous self-play episodes={} episode_offset={} resume={} dt={:.3} max_steps={} player_armed={} zombie_params={} player_params={}",
        episodes,
        episode_offset,
        resume,
        dt(),
        max_steps(),
        loadout_probe.armed,
        zombie_policy.params(),
        player_policy.params()
    );

    for episode in 1..=episodes {
        let global_episode = episode_offset + episode;
        let train_player = true;
        let train_zombies = true;
        let metrics = run_episode(
            &mut zombie_policy,
            &mut player_policy,
            &mut rng,
            epsilon_zombie,
            epsilon_player,
            false,
            train_zombies,
            train_player,
            lr,
            gamma,
        );

        epsilon_zombie = (epsilon_zombie * epsilon_decay).max(epsilon_min);
        epsilon_player = (epsilon_player * epsilon_decay).max(epsilon_min);

        if episode % 100 == 0 {
            let path = format!("{}/replays/replay_ep_{:05}.csv", output_dir, global_episode);
            write_replay_sample(Path::new(&path), global_episode, &metrics);
        }
        if episode % 500 == 0 {
            let summary = evaluate(
                &mut zombie_policy,
                &mut player_policy,
                seed + global_episode as u64,
                64,
                false,
            );
            let phase = "simultaneous_selfplay";
            let score = eval_score(&summary);
            if score > best_score {
                best_score = score;
                best_episode = global_episode;
                best_zombie_policy = zombie_policy.clone();
                best_player_policy = player_policy.clone();
                write_checkpoint(
                    Path::new(&format!("{}/zombie_policy_best.json", output_dir)),
                    &best_zombie_policy,
                );
                write_checkpoint(
                    Path::new(&format!("{}/player_policy_best.json", output_dir)),
                    &best_player_policy,
                );
            } else {
                zombie_policy = best_zombie_policy.clone();
                player_policy = best_player_policy.clone();
            }
            println!(
                "eval ep={} phase={} zombie_win={:.3} mean_ttk={:.3}s z_reward={:.3} z_entropy={:.3} score={:.3} best_ep={} eps_z={:.3} eps_p={:.3}",
                global_episode,
                phase,
                summary.zombie_win_rate,
                summary.mean_ttk,
                summary.zombie_reward,
                summary.zombie_action_entropy,
                score,
                best_episode,
                epsilon_zombie,
                epsilon_player
            );
            println!(
                "zombie_actions {}",
                action_histogram(&ZOMBIE_ACTIONS, &summary.zombie_action_counts)
            );
            println!(
                "player_actions {}",
                action_histogram(&PLAYER_ACTIONS, &summary.player_action_counts)
            );
            report.push(format!(
                "{},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{},{},{},{:.5},{}",
                global_episode,
                phase,
                epsilon_zombie,
                epsilon_player,
                summary.zombie_win_rate,
                summary.mean_ttk,
                summary.zombie_reward,
                summary.zombie_action_entropy,
                summary.player_action_entropy,
                score,
                zombie_policy.params(),
                player_policy.params(),
                best_episode,
                dt(),
                max_steps()
            ));
            write_report(Path::new(&report_path), &report);
        }
        if episode % 1000 == 0 {
            write_checkpoint(
                Path::new(&format!(
                    "{}/zombie_policy_ep_{:05}.json",
                    output_dir, global_episode
                )),
                &zombie_policy,
            );
            write_checkpoint(
                Path::new(&format!(
                    "{}/player_policy_ep_{:05}.json",
                    output_dir, global_episode
                )),
                &player_policy,
            );
        }
    }

    write_checkpoint(
        Path::new("checkpoints/zombie_policy.json"),
        &best_zombie_policy,
    );
    write_checkpoint(
        Path::new("checkpoints/zombie_policy_attention_release.json"),
        &best_zombie_policy,
    );
    write_checkpoint(
        Path::new("checkpoints/player_policy.json"),
        &best_player_policy,
    );
    write_checkpoint(
        Path::new(&format!(
            "{}/zombie_policy_best_ep_{:05}.json",
            output_dir, best_episode
        )),
        &best_zombie_policy,
    );
    write_checkpoint(
        Path::new(&format!(
            "{}/player_policy_best_ep_{:05}.json",
            output_dir, best_episode
        )),
        &best_player_policy,
    );
    write_checkpoint(
        Path::new(&format!("{}/zombie_policy_final.json", output_dir)),
        &zombie_policy,
    );
    write_checkpoint(
        Path::new(&format!("{}/player_policy_final.json", output_dir)),
        &player_policy,
    );
    write_report(Path::new(&report_path), &report);
}

fn live_zombie_indices(zombies: &[Zombie]) -> Vec<usize> {
    zombies
        .iter()
        .enumerate()
        .filter_map(|(idx, zombie)| if zombie.hp > 0.0 { Some(idx) } else { None })
        .collect()
}

fn nearest_zombie(player: &Player, zombies: &[Zombie], live: &[usize]) -> Option<usize> {
    live.iter().copied().min_by(|left, right| {
        dist(player.x, player.y, zombies[*left].x, zombies[*left].y)
            .partial_cmp(&dist(
                player.x,
                player.y,
                zombies[*right].x,
                zombies[*right].y,
            ))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn mean_distance(player: &Player, zombies: &[Zombie], live: &[usize]) -> f32 {
    if live.is_empty() {
        return 0.0;
    }
    live.iter()
        .map(|idx| dist(player.x, player.y, zombies[*idx].x, zombies[*idx].y))
        .sum::<f32>()
        / live.len() as f32
}

fn is_surrounded(player: &Player, zombies: &[Zombie]) -> bool {
    let mut sectors = [false; 4];
    for zombie in zombies.iter().filter(|zombie| zombie.hp > 0.0) {
        if dist(player.x, player.y, zombie.x, zombie.y) > 48.0 {
            continue;
        }
        let dx = zombie.x - player.x;
        let dy = zombie.y - player.y;
        let idx = if dx.abs() > dy.abs() {
            if dx > 0.0 {
                0
            } else {
                1
            }
        } else if dy > 0.0 {
            2
        } else {
            3
        };
        sectors[idx] = true;
    }
    sectors.iter().filter(|value| **value).count() >= 3
}

fn move_actor(
    x: &mut f32,
    y: &mut f32,
    dir: (f32, f32),
    amount: f32,
    stuck_x: &mut f32,
    stuck_y: &mut f32,
) {
    let old = (*x, *y);
    *x += dir.0 * amount;
    *y += dir.1 * amount;
    *stuck_x = 0.0;
    *stuck_y = 0.0;
    if *x < -120.0 {
        *x = -120.0;
        *stuck_x = -1.0;
    }
    if *x > 120.0 {
        *x = 120.0;
        *stuck_x = 1.0;
    }
    if *y < -120.0 {
        *y = -120.0;
        *stuck_y = -1.0;
    }
    if *y > 120.0 {
        *y = 120.0;
        *stuck_y = 1.0;
    }
    if dist(old.0, old.1, *x, *y) > 0.01 {
        let face = normalized(*x - old.0, *y - old.1);
        if stuck_x.abs() < 0.5 {
            *stuck_x += face.0 * 0.0;
        }
    }
}

fn dist(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

fn normalized(x: f32, y: f32) -> (f32, f32) {
    let len = (x * x + y * y).sqrt();
    if len <= 0.001 {
        (1.0, 0.0)
    } else {
        (x / len, y / len)
    }
}

fn can_attend(query_idx: usize, key_idx: usize, prefix_len: usize) -> bool {
    if query_idx < prefix_len {
        key_idx < prefix_len
    } else {
        key_idx < prefix_len || key_idx <= query_idx
    }
}

fn relative_position_bias(query_idx: usize, key_idx: usize) -> f32 {
    -0.015 * (query_idx as f32 - key_idx as f32).abs()
}

fn type_embedding(entity_type: usize, dim: usize, d_model: usize) -> f32 {
    deterministic_weight(
        240_000 + (entity_type * d_model + dim) as u64,
        (d_model as f32).sqrt(),
    )
}

fn scale_token(token: &mut [f32], denom: usize) {
    let scale = 1.0 / (denom.max(1) as f32).sqrt();
    for value in token {
        *value *= scale;
    }
}

fn deterministic_matrix(rows: usize, cols: usize, seed: u64, fan_scale: f32) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| deterministic_weight(seed + (row * cols + col) as u64, fan_scale))
                .collect()
        })
        .collect()
}

fn deterministic_weight(index: u64, fan_scale: f32) -> f32 {
    let mut x = index.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    let unit = x as f64 / u64::MAX as f64;
    ((unit * 2.0 - 1.0) as f32) / fan_scale.max(1.0)
}

fn configure_sim_timing(sim_dt: f32, episode_seconds: f32) {
    let _ = SIM_DT.set(sim_dt.clamp(0.01, 0.5));
    let _ = SIM_EPISODE_SECONDS.set(episode_seconds.clamp(1.0, 600.0));
}

fn dt() -> f32 {
    SIM_DT.get().copied().unwrap_or(DEFAULT_DT)
}

fn episode_seconds() -> f32 {
    SIM_EPISODE_SECONDS
        .get()
        .copied()
        .unwrap_or(DEFAULT_EPISODE_SECONDS)
}

fn max_steps() -> usize {
    (episode_seconds() / dt()).round().max(1.0) as usize
}

fn mat_vec(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; matrix.len()];
    for (row_idx, row) in matrix.iter().enumerate() {
        output[row_idx] = dot_product(row, vector);
    }
    output
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    let left = &left[..len];
    let right = &right[..len];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { dot_product_avx(left, right) };
        }
    }

    dot_product_scalar(left, right)
}

fn dot_product_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_avx(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut idx = 0usize;
    let mut sum = _mm256_setzero_ps();
    while idx + 8 <= left.len() {
        let a = unsafe { _mm256_loadu_ps(left.as_ptr().add(idx)) };
        let b = unsafe { _mm256_loadu_ps(right.as_ptr().add(idx)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
        idx += 8;
    }

    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut total = lanes.iter().sum::<f32>();
    while idx < left.len() {
        total += left[idx] * right[idx];
        idx += 1;
    }
    total
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_avx(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86::*;

    let mut idx = 0usize;
    let mut sum = _mm256_setzero_ps();
    while idx + 8 <= left.len() {
        let a = unsafe { _mm256_loadu_ps(left.as_ptr().add(idx)) };
        let b = unsafe { _mm256_loadu_ps(right.as_ptr().add(idx)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
        idx += 8;
    }

    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut total = lanes.iter().sum::<f32>();
    while idx < left.len() {
        total += left[idx] * right[idx];
        idx += 1;
    }
    total
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

fn apply_rope(values: &mut [f32], heads: usize, head_dim: usize, position: usize) {
    if heads == 0 || head_dim < 2 {
        return;
    }
    for head_idx in 0..heads {
        let head_offset = head_idx * head_dim;
        for pair_idx in 0..(head_dim / 2) {
            let even_idx = head_offset + pair_idx * 2;
            let odd_idx = even_idx + 1;
            if odd_idx >= values.len() {
                break;
            }
            let theta =
                position as f32 / 10_000.0_f32.powf((pair_idx * 2) as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let even = values[even_idx];
            let odd = values[odd_idx];
            values[even_idx] = even * cos - odd * sin;
            values[odd_idx] = even * sin + odd * cos;
        }
    }
}

fn flash_d_recurrence_weight(score: f32, previous_score: f32, previous_weight: f32) -> f32 {
    sigmoid_stable(score - previous_score + previous_weight.max(1.0e-6).ln())
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

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_idx = idx;
        }
    }
    best_idx
}

fn parse_arg<T: std::str::FromStr>(name: &str, default_value: T) -> T {
    let prefix = format!("--{}=", name);
    std::env::args()
        .find_map(|arg| {
            arg.strip_prefix(&prefix)
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(default_value)
}

fn parse_bool_arg(name: &str, default_value: bool) -> bool {
    let prefix = format!("--{}=", name);
    std::env::args()
        .find_map(|arg| {
            arg.strip_prefix(&prefix).map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
        .unwrap_or(default_value)
}

fn decayed_epsilon(start: f32, decay: f32, min_value: f32, train_count: usize) -> f32 {
    (start * decay.powi(train_count as i32)).max(min_value)
}

#[derive(Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xA5A5_A5A5_5A5A_5A5A,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = self.state;
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    fn f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.f32()
    }

    fn usize(&mut self, low: usize, high: usize) -> usize {
        low + (self.next_u64() as usize % (high - low + 1))
    }
}
