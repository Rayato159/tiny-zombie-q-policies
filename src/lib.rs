pub const FEATURE_COUNT: usize = 23;
pub const PLAYER_FEATURE_COUNT: usize = 15;
pub const ACTION_COUNT: usize = 5;
pub const PARAMETER_BUDGET: usize = 10_000;

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "is_player_armed",
    "is_player_stamina_less_half",
    "is_player_health_less_half",
    "is_player_stuck",
    "player_stuck_normal_x",
    "player_stuck_normal_y",
    "nearby_zombie_count",
    "dash_ready",
    "attack_ready",
    "nearest_zombie_distance",
    "nearest_zombie_dir_x",
    "nearest_zombie_dir_y",
    "player_speed",
    "swarm_centroid_dir_x",
    "swarm_centroid_dir_y",
    "player_facing_dot_nearest_zombie",
    "nearest_zombie_side_sign",
    "backstab_opportunity",
    "swarm_left_pressure",
    "swarm_right_pressure",
    "swarm_front_pressure",
    "swarm_back_pressure",
    "swarm_spread",
];

pub const PLAYER_FEATURE_NAMES: [&str; PLAYER_FEATURE_COUNT] = [
    "health_ratio",
    "stamina_ratio",
    "is_player_armed",
    "is_player_attacking",
    "nearest_zombie_distance",
    "nearest_zombie_dir_x",
    "nearest_zombie_dir_y",
    "nearest_zombie_attacking",
    "zombie_count",
    "pressure_count",
    "player_speed",
    "player_facing_dot_nearest_zombie",
    "swarm_centroid_dir_x",
    "swarm_centroid_dir_y",
    "dodge_ready",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TacticalAction {
    Attack,
    FlankLeft,
    FlankRight,
    DashIn,
    DashOut,
}

impl TacticalAction {
    pub const ALL: [Self; ACTION_COUNT] = [
        Self::Attack,
        Self::FlankLeft,
        Self::FlankRight,
        Self::DashIn,
        Self::DashOut,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub actions: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            input_dim: FEATURE_COUNT,
            hidden_dim: 34,
            actions: ACTION_COUNT,
        }
    }
}

impl MlpConfig {
    pub fn parameter_count(self) -> usize {
        self.hidden_dim * self.input_dim
            + self.hidden_dim
            + self.actions * self.hidden_dim
            + self.actions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedAttentionConfig {
    pub input_dim: usize,
    pub d_model: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub actions: usize,
}

impl Default for GroupedAttentionConfig {
    fn default() -> Self {
        Self {
            input_dim: FEATURE_COUNT,
            d_model: 24,
            layers: 2,
            query_heads: 4,
            kv_heads: 1,
            actions: ACTION_COUNT,
        }
    }
}

impl GroupedAttentionConfig {
    pub fn parameter_count(self) -> usize {
        assert!(self.query_heads > 0);
        assert!(self.kv_heads > 0);
        assert_eq!(self.d_model % self.query_heads, 0);
        let head_dim = self.d_model / self.query_heads;
        let kv_dim = self.kv_heads * head_dim;

        let feature_params = self.input_dim * self.d_model;
        let token_params = self.d_model * 2;
        let per_layer_params =
            self.d_model + 2 * self.d_model * self.d_model + 2 * kv_dim * self.d_model;
        let output_params = self.d_model + self.actions * self.d_model + self.actions;

        feature_params + token_params + self.layers * per_layer_params + output_params
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalSummary {
    pub mode: String,
    pub player_death_rate: f32,
    pub mean_reward_per_episode: f32,
    pub mean_player_damage_per_episode: f32,
    pub action_diversity: f32,
}

pub fn choose_empirical_winner<'a>(summaries: &'a [EvalSummary]) -> Option<&'a EvalSummary> {
    summaries.iter().max_by(|left, right| {
        left.player_death_rate
            .total_cmp(&right.player_death_rate)
            .then_with(|| {
                left.mean_reward_per_episode
                    .total_cmp(&right.mean_reward_per_episode)
            })
            .then_with(|| {
                left.mean_player_damage_per_episode
                    .total_cmp(&right.mean_player_damage_per_episode)
            })
            .then_with(|| left.action_diversity.total_cmp(&right.action_diversity))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_models_stay_under_budget() {
        assert_eq!(MlpConfig::default().parameter_count(), 991);
        assert_eq!(GroupedAttentionConfig::default().parameter_count(), 3677);
        assert!(MlpConfig::default().parameter_count() <= PARAMETER_BUDGET);
        assert!(GroupedAttentionConfig::default().parameter_count() <= PARAMETER_BUDGET);
    }

    #[test]
    fn empirical_winner_uses_reward_after_death_rate() {
        let summaries = [
            EvalSummary {
                mode: "Rule".to_string(),
                player_death_rate: 1.0,
                mean_reward_per_episode: 9.83,
                mean_player_damage_per_episode: 144.0,
                action_diversity: 0.9361,
            },
            EvalSummary {
                mode: "Attention".to_string(),
                player_death_rate: 1.0,
                mean_reward_per_episode: 28.77,
                mean_player_damage_per_episode: 135.0,
                action_diversity: 0.576,
            },
        ];
        let winner = choose_empirical_winner(&summaries).unwrap();
        assert_eq!(winner.mode, "Attention");
    }

    #[test]
    fn empirical_winner_can_still_be_rule() {
        let summaries = [
            EvalSummary {
                mode: "Rule".to_string(),
                player_death_rate: 1.0,
                mean_reward_per_episode: 31.0,
                mean_player_damage_per_episode: 144.0,
                action_diversity: 0.9361,
            },
            EvalSummary {
                mode: "Attention".to_string(),
                player_death_rate: 1.0,
                mean_reward_per_episode: 28.77,
                mean_player_damage_per_episode: 135.0,
                action_diversity: 0.576,
            },
        ];
        let winner = choose_empirical_winner(&summaries).unwrap();
        assert_eq!(winner.mode, "Rule");
    }
}
