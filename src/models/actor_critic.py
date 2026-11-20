"""
Actor-Critic Neural Network Architecture
Responsible: SV3 (Mạng + Đánh giá + Điều phối)
Milestone deadline: 18/11 - 19/11 (Review & test: 20/11)
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    """
    ActorCritic Model using Nature-DQN style CNN feature extractor (T4 18/11 & T5 19/11 - SV3).
    Input shape: (Batch, In_Channels, 84, 84) - default In_Channels = 1 (grayscale) or 4 (frame stack).
    """
    def __init__(self, in_channels: int = 1, num_actions: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        # Feature extractor: Convolutional Backbone
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute conv output size for 84x84 input: 64 * 7 * 7 = 3136
        conv_out_size = 64 * 7 * 7

        # Shared representation layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        # Actor head: Outputs logits for discrete action distribution
        self.actor = nn.Linear(512, num_actions)

        # Critic head: Outputs scalar state value V(s)
        self.critic = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal weight initialization for stable PPO training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Small gain for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        # Standard gain for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (T4 18/11 - SV3).
        Returns:
            action_logits: Tensor of shape (Batch, num_actions)
            values: Tensor of shape (Batch, 1)
        """
        features = self.conv(x)
        hidden = self.fc(features)
        logits = self.actor(hidden)
        values = self.critic(hidden)
        return logits, values

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects action during rollout or evaluates existing action.
        Returns:
            action: Selected action tensor.
            log_prob: Log probability of the chosen action.
            entropy: Entropy of policy distribution.
            value: Estimated state value V(s).
        """
        logits, values = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, values.squeeze(-1)

    def evaluate_actions(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates batch of state-action pairs during PPO update.
        """
        logits, values = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, values.squeeze(-1), entropy
