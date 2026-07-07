"""
Phase 2 验证: UAV蜂群模型 + 威胁/效益因子 + DRTA模型
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env.uav_model import UAVSwarmDensityModel, UAVKinematics, generate_uav_configs
from env.physics import ThreatBenefitFactors, DRTAModel, HELSDamageModel


def test_swarm_arrival_times():
    """验证蜂群到达时间分布"""
    print("\n--- Test 1: Swarm Arrival Times ---")
    model = UAVSwarmDensityModel(mode='multi_batch')
    times = model.generate_arrival_times(50, scenario_duration=300.0)
    print(f"  Generated {len(times)} arrival times (requested 50)")
    print(f"  Range: [{times[0]:.1f}, {times[-1]:.1f}] s")
    print(f"  Mean interval: {np.mean(np.diff(times)):.1f} s")
    assert len(times) >= 40, "Should generate ~50 arrivals"
    assert times[0] > 0, "First arrival should be > 0"
    print("  [PASS]")


def test_uav_kinematics():
    """验证UAV运动学"""
    print("\n--- Test 2: UAV Kinematics ---")
    uav = UAVKinematics(0, (10000, 0, 500), speed=25.0, target_pos=(0, 0, 0))
    assert uav.alive
    assert uav.velocity[0] < 0, "Should move toward origin"

    # Step for 100 seconds
    for _ in range(100):
        uav.step(1.0)

    dist = uav.distance_to((0, 0, 0))
    expected_dist = np.sqrt((10000 - 25 * 100) ** 2 + 0 ** 2 + 500 ** 2)
    assert abs(dist - expected_dist) < 1.0, f"Distance {dist:.1f} != expected {expected_dist:.1f}"
    print(f"  After 100s: position=({uav.position[0]:.0f}, {uav.position[1]:.0f}, {uav.position[2]:.0f})")
    print(f"  Distance to target: {dist:.1f} m (expected {expected_dist:.1f} m)")
    print("  [PASS]")


def test_uav_config_generation():
    """验证UAV配置生成"""
    print("\n--- Test 3: UAV Config Generation ---")
    rng = np.random.RandomState(42)
    arrival = np.linspace(0, 200, 50)
    configs = generate_uav_configs(50, arrival, rng=rng)

    heights = [u.height for u in configs]
    speeds = [u.speed for u in configs]
    print(f"  Generated {len(configs)} UAV configs")
    print(f"  Height range: [{min(heights):.0f}, {max(heights):.0f}] m")
    print(f"  Speed range: [{min(speeds):.1f}, {max(speeds):.1f}] m/s")

    for u in configs[:3]:
        dist = u.distance_to((0, 0, 0))
        print(f"  UAV {u.id}: pos=({u.position[0]:.0f},{u.position[1]:.0f},{u.position[2]:.0f}), "
              f"dist={dist:.0f}m, speed={u.speed:.1f}m/s")

    assert 200 <= min(heights) <= 800
    assert 200 <= max(heights) <= 800
    assert 20 <= min(speeds) <= 30
    print("  [PASS]")


def test_threat_benefit_integration():
    """验证威胁因子与雷达/环境参数的集成"""
    print("\n--- Test 4: Threat-Benefit Integration ---")
    model = HELSDamageModel('rural', 'Al2024')
    tbf = ThreatBenefitFactors()

    # Simulate a UAV at 5km, 500m altitude, 25m/s
    L, h, v = 5000.0, 500.0, 25.0
    theta = np.arctan2(h, L)
    result = model.compute_damage_time(30e3, L, h, theta)

    L_remaining = L  # distance to protected asset
    factors = tbf.compute_all(h, v, L_remaining, result['t_damage'],
                               30e3, 50e3, 150.0, 200.0)

    print(f"  UAV at {L/1000:.0f}km, h={h:.0f}m, v={v:.0f}m/s")
    print(f"  t_damage={result['t_damage']:.3f}s, tau={result['tau']:.4f}")
    print(f"  T_rh={factors[0]:.3f}, T_rv={factors[1]:.3f}, T_rL={factors[2]:.3f}")
    print(f"  B_rc={factors[3]:.3f}, B_rs={factors[4]:.3f}")
    print("  [PASS]")


def test_drta_objective():
    """验证DRTA目标函数一致性"""
    print("\n--- Test 5: DRTA Objective Range ---")
    dmodel = DRTAModel()
    tbf = ThreatBenefitFactors()

    # Test across a range of UAV states
    scenarios = [
        (200, 25, 100, 1.0, 30e3, 150),   # close, fast, ideal speed
        (800, 15, 5000, 10.0, 50e3, 50),  # far, slow, long damage
        (500, 25, 500, 2.0, 30e3, 100),   # mid-range
    ]
    for h, v, Lr, td, P0, bat in scenarios:
        f = tbf.compute_all(h, v, Lr, td, P0, 50e3, bat, 200.0)
        obj = dmodel.objective(*f)
        assert 0 <= obj <= 1, f"Objective {obj:.4f} out of [0,1]"
        print(f"  h={h:.0f}m v={v:.0f}m/s Lr={Lr:.0f}m td={td:.1f}s -> obj={obj:.4f}")

    print("  [PASS]")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2: UAV Model + Threat/Benefit Factors Verification")
    print("=" * 60)
    test_swarm_arrival_times()
    test_uav_kinematics()
    test_uav_config_generation()
    test_threat_benefit_integration()
    test_drta_objective()
    print("\n" + "=" * 60)
    print("Phase 2: ALL TESTS PASSED")
    print("=" * 60)
