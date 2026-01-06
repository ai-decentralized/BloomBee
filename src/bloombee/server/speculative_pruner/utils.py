from dataclasses import dataclass
from typing import List

@dataclass
class NetworkCondition:
    """Network condition metrics"""
    latency: float  # ms
    bandwidth: float  # Mbps
    packet_loss: float  # 0-1
    throughput: float  # Current throughput Mbps
    
    def to_features(self) -> List[float]:
        """Convert to normalized features for neural network"""
        return [
            min(self.latency / 200.0, 1.0),  # Normalize to [0, 1]
            min(self.bandwidth / 100.0, 1.0),
            self.packet_loss,
            min(self.throughput / 100.0, 1.0)
        ]
    
    @classmethod
    def mock(cls, condition_type: str = 'normal'):
        """Mock network condition for training"""
        conditions = {
            'good': {'latency': 10, 'bandwidth': 500, 'packet_loss': 0.001, 'throughput': 90},
            'normal': {'latency': 50, 'bandwidth': 100, 'packet_loss': 0.01, 'throughput': 40},
            'poor': {'latency': 150, 'bandwidth': 20, 'packet_loss': 0.05, 'throughput': 15}
        }
        
        base = conditions.get(condition_type, conditions['poor'])
        return cls(
            latency=base['latency'],
            bandwidth=base['bandwidth'],
            packet_loss=base['packet_loss'],
            throughput=base['throughput'],
        )
        # Add some randomness
        # return cls(
        #     latency=base['latency'] * (1 + random.uniform(-0.2, 0.2)),
        #     bandwidth=base['bandwidth'] * (1 + random.uniform(-0.1, 0.1)),
        #     packet_loss=base['packet_loss'] * (1 + random.uniform(-0.3, 0.3)),
        #     throughput=base['throughput'] * (1 + random.uniform(-0.15, 0.15))
        # )
        
        
        