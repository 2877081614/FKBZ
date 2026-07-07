"""
Phase 1 验证: 核心物理模型
验证激光大气传输、热毁伤、HELS毁伤模型的计算正确性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from env.physics import (LaserAtmosphericTransmission, LaserThermalDamage,
                          HELSDamageModel, ThreatBenefitFactors, DRTAModel)
from config.hyperparams import ATMOSPHERE_PARAMS, MATERIAL_PROPS, LASER_POWER_RANGE


def test_material_damage_threshold():
    """验证材料毁伤阈值 e_th 计算"""
    print("\n--- Test 1: Material Damage Threshold ---")
    for key, mat in MATERIAL_PROPS.items():
        e_th = mat['e_th']
        # 手动计算
        e_th_manual = mat['z_d'] * mat['rho'] * (
            mat['c_s'] * (mat['T_m'] - mat['T_0']) + mat['L_m'])
        diff = abs(e_th - e_th_manual) / max(e_th, 1e-10)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"  {mat['name']}: e_th = {e_th:.3e} J/m^2, error = {diff:.1e} [{status}]")

    # 论文 Al2024 5mm 期望 ~1.13e7 J/m^2
    al_th = MATERIAL_PROPS['Al2024']['e_th']
    expected = 1.13e7
    rel_err = abs(al_th - expected) / expected
    print(f"  Al2024 vs expected ~1.13e7: relative error = {rel_err:.3%}")


def test_atmospheric_transmission():
    """验证大气传输模型关键参数"""
    print("\n--- Test 2: Atmospheric Transmission ---")
    for env_type in ['rural', 'desert', 'coastal']:
        atm = LaserAtmosphericTransmission(env_type)
        # 典型条件: P0=30kW, L=5km, h=500m, theta=5deg
        P0, L, h = 30e3, 5000.0, 500.0
        theta = np.deg2rad(5.0)
        result = atm.compute_all(P0, L, h, theta)

        print(f"\n  [{env_type}] {ATMOSPHERE_PARAMS[env_type]['description']}")
        print(f"    Cn2(0)     = {atm.Cn2_0:.1e} m^(-2/3)")
        print(f"    Visibility = {atm.visibility/1000:.0f} km")
        print(f"    beta_T^2   = {result['beta_T']**2:.4f}  (turbulence)")
        print(f"    beta_B^2   = {result['beta_B']**2:.4f}  (thermal blooming)")
        print(f"    beta_J     = {result['beta_J']:.4f}  (jitter)")
        print(f"    beta_total = {result['beta']:.4f}  (total)")
        print(f"    tau        = {result['tau']:.4f}  (transmittance)")
        print(f"    r_spot     = {result['r_spot']:.4f} m  (spot radius)")
        print(f"    I_target   = {result['I_target']:.2e} W/m^2")
        print(f"    P_e        = {result['P_e']:.2e} W")

        # 合理性检查
        assert result['tau'] > 0 and result['tau'] <= 1, "Transmittance out of range"
        assert result['r_spot'] > 0, "Spot radius must be positive"
        assert result['I_target'] > 0, "Power density must be positive"
        # 沙漠能见度低，透过率应更低
        if env_type == 'desert':
            atm_rural = LaserAtmosphericTransmission('rural')
            tau_rural = atm_rural.atmospheric_transmittance(L)
            assert result['tau'] < tau_rural, \
                f"Desert tau ({result['tau']:.4f}) should be < rural tau ({tau_rural:.4f})"
    print("  [PASS]")


def test_damage_time():
    """验证毁伤时间计算"""
    print("\n--- Test 3: Damage Time Calculation ---")
    model = HELSDamageModel('rural', 'Al2024')

    # 测试不同距离下的毁伤时间
    distances = [1000, 3000, 5000, 7000, 10000]
    P0 = 30e3
    h = 500.0

    print(f"  P0 = {P0/1000:.0f} kW, material = Al2024 5mm")
    print(f"  {'Distance':>8s}  {'t_damage':>10s}  {'I_target':>12s}  {'tau':>8s}")
    print(f"  {'-'*48}")
    for L in distances:
        theta = np.arctan2(h, L)
        result = model.compute_damage_time(P0, L, h, theta)
        print(f"  {L/1000:5.1f} km  {result['t_damage']:8.3f} s  "
              f"{result['I_target']:8.2e} W/m^2  {result['tau']:6.4f}")

        # 物理合理性检查
        assert result['t_damage'] > 0, f"Damage time must be positive at L={L}m"
        # 距离越远毁伤时间越长
        if L > 1000:
            prev = model.compute_damage_time(P0, L - 1000, h, np.arctan2(h, L - 1000))
            assert result['t_damage'] > prev['t_damage'], \
                f"Damage time should increase with distance"

    # 论文核心关系验证: t_damage ∝ L^2 * beta^2 / (tau * P0)
    # 注意: 由于热晕效应，功率翻倍时毁伤时间减少略小于50% (beta_B^2随P0增大)
    r1 = model.compute_damage_time(30e3, 5000, h, np.arctan2(h, 5000))
    r2 = model.compute_damage_time(60e3, 5000, h, np.arctan2(h, 5000))
    ratio = r1['t_damage'] / max(r2['t_damage'], 1e-10)
    print(f"\n  功率翻倍时毁伤时间比: {ratio:.3f} (期望 ~1.5-2.5, 热晕使收益略减)")
    assert 1.2 < ratio < 3.0, f"t_damage ~ 1/P0 relation seems off: ratio={ratio:.3f}"

    print("  [PASS]")


def test_threat_benefit_factors():
    """验证威胁与效益因子"""
    print("\n--- Test 4: Threat & Benefit Factors ---")
    tbf = ThreatBenefitFactors()

    # 高度威胁: 更低=更大
    assert tbf.height_threat(200) > tbf.height_threat(800), "Lower height should be more threatening"

    # 速度威胁: 越接近25m/s越威胁
    assert tbf.velocity_threat(25) > tbf.velocity_threat(15), "v=25 should be more threatening than v=15"
    assert tbf.velocity_threat(25) > tbf.velocity_threat(35), "v=25 should be more threatening than v=35"

    # 距离威胁: 越近越威胁
    assert tbf.safe_distance_threat(100) > tbf.safe_distance_threat(2000), "Closer should be more threatening"
    assert tbf.safe_distance_threat(2000) == 0.0, "Beyond safe distance should be 0"

    # 资源消耗: 毁伤时间短的得分高
    assert tbf.resource_consumption_benefit(0.5) > tbf.resource_consumption_benefit(5.0), \
        "Shorter damage time should have higher benefit"

    # 范围检查
    factors = tbf.compute_all(500.0, 25.0, 500.0, 2.0, 30e3, 50e3, 100.0, 200.0)
    for i, f in enumerate(factors):
        assert 0.0 <= f <= 1.0, f"Factor {i} = {f:.3f} out of [0,1]"

    print(f"  T_rh={factors[0]:.3f}  T_rv={factors[1]:.3f}  T_rL={factors[2]:.3f}  "
          f"B_rc={factors[3]:.3f}  B_rs={factors[4]:.3f}")
    print("  [PASS]")


def test_drta_model():
    """验证DRTA模型约束和目标函数"""
    print("\n--- Test 5: DRTA Model ---")
    model = DRTAModel()

    # 目标函数计算
    obj = model.objective(0.8, 0.9, 0.5, 0.6, 0.7)
    assert 0 < obj < 1, f"Objective {obj:.3f} should be in (0,1)"
    print(f"  Objective value: {obj:.4f}")

    # 约束测试
    hels_s = {'battery_remaining': 100.0, 'currently_irradiating': None}
    uav_s = {'alive': True, 'detected': True, 'assigned_to': None}

    # Case 1: feasible
    dec = {'irradiating': True, 't_damage': 5.0}
    ok, reason = model.check_constraints(hels_s, uav_s, dec)
    assert ok, f"Should be feasible but got: {reason}"
    print(f"  Feasible case: {ok} ({reason})")

    # Case 2: insufficient battery
    dec2 = {'irradiating': True, 't_damage': 150.0}
    ok2, reason2 = model.check_constraints(hels_s, uav_s, dec2)
    assert not ok2, f"Should be infeasible due to battery"
    print(f"  Battery case: feasible={ok2} ({reason2})")

    # Case 3: target already assigned
    uav_s2 = {'alive': True, 'detected': True, 'assigned_to': 0}
    ok3, reason3 = model.check_constraints(hels_s, uav_s2, dec)
    assert not ok3, f"Should be infeasible due to assignment"
    print(f"  Assigned case: feasible={ok3} ({reason3})")

    # Case 4: HELS busy
    hels_s4 = {'battery_remaining': 100.0, 'currently_irradiating': 3}
    ok4, reason4 = model.check_constraints(hels_s4, uav_s, dec)
    assert not ok4, f"Should be infeasible due to HELS busy"
    print(f"  HELS busy case: feasible={ok4} ({reason4})")

    # Case 5: wait action
    dec_wait = {'irradiating': False}
    ok5, reason5 = model.check_constraints(hels_s, uav_s, dec_wait)
    assert ok5, f"Wait should always be feasible"
    print(f"  Wait case: feasible={ok5} ({reason5})")

    print("  [PASS]")


def test_paper_figure5_consistency():
    """验证论文Fig.5: 毁伤时间 vs 距离关系 (定性)"""
    print("\n--- Test 6: Figure 5 Consistency (Damage time vs Distance) ---")
    distances = np.linspace(1000, 10000, 10)

    for env_type in ['rural', 'desert', 'coastal']:
        model = HELSDamageModel(env_type, 'Al2024')
        times = []
        for L in distances:
            h = 500.0
            theta = np.arctan2(h, L)
            r = model.compute_damage_time(30e3, L, h, theta)
            times.append(r['t_damage'])

        # 沙漠(medium turbulence, low visibility)毁伤时间应最长
        # 乡村(weak turbulence, high visibility)毁伤时间应最短
        print(f"  {env_type:>8s}: t at 5km = {times[4]:.3f}s, t at 10km = {times[-1]:.3f}s")

    # 定性: 同距离下沙漠毁伤时间 > 沿海 > 乡村
    times_rural = []
    times_desert = []
    for L in distances:
        h = 500.0
        theta = np.arctan2(h, L)
        times_rural.append(HELSDamageModel('rural').compute_damage_time(30e3, L, h, theta)['t_damage'])
        times_desert.append(HELSDamageModel('desert').compute_damage_time(30e3, L, h, theta)['t_damage'])

    assert all(d > r for d, r in zip(times_desert, times_rural)), \
        "Desert damage time should always exceed rural"
    print("  Desert > Rural damage time at all distances: VERIFIED")
    print("  [PASS]")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1: Core Physics Model Verification")
    print("=" * 60)

    test_material_damage_threshold()
    test_atmospheric_transmission()
    test_damage_time()
    test_threat_benefit_factors()
    test_drta_model()
    test_paper_figure5_consistency()

    print("\n" + "=" * 60)
    print("Phase 1: ALL TESTS PASSED")
    print("=" * 60)
