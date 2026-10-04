"""
游戏内生态系统与生物AI系统 - 验收测试
运行方式：python -m pytest tests/test_ecosystem.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.ecosystem import Ecosystem, AI_STATES


class TestEcosystem:
    def setup_method(self):
        self.system = Ecosystem()

    # ---- 生物AI：状态机 / 感知 / 觅食 / 躲避 / 繁殖 ----

    def test_case_01(self):
        """AI有状态机：creature_ai 返回合法状态"""
        rabbit = self.system.spawn_creature("rabbit", 0, 0)
        state = self.system.creature_ai(rabbit)
        assert state in AI_STATES
        assert rabbit.state == state

    def test_case_02(self):
        """AI有感知：只能看到感知半径内的食物"""
        rabbit = self.system.spawn_creature("rabbit", 0, 0, energy=1.0)
        self.system.spawn_creature("grass", 5, 0)      # 感知半径内
        self.system.spawn_creature("grass", 1000, 0)   # 感知半径外
        state = self.system.creature_ai(rabbit)
        assert state == "forage"
        # 远处的草不在感知范围内，最近目标应是近处那棵
        near = min((c for c in self.system.creatures if c.species == "grass"),
                   key=lambda g: rabbit.distance_to(g))
        assert near.x == 5

    def test_case_03(self):
        """AI会躲避：感知到天敌后进入 flee 并远离天敌"""
        rabbit = self.system.spawn_creature("rabbit", 0, 0)
        wolf = self.system.spawn_creature("wolf", 5, 0)
        before = rabbit.distance_to(wolf)
        state = self.system.creature_ai(rabbit, dt=1.0)
        assert state == "flee"
        assert rabbit.distance_to(wolf) > before

    def test_case_04(self):
        """AI会觅食：饥饿时吃掉食物并增加能量"""
        rabbit = self.system.spawn_creature("rabbit", 0, 0, energy=1.0)
        grass = self.system.spawn_creature("grass", 0.5, 0)
        state = self.system.creature_ai(rabbit)
        assert state == "forage"
        assert grass not in self.system.creatures
        assert rabbit.energy > 1.0

    def test_case_05(self):
        """AI会繁殖：能量充足时产生后代"""
        before = len(self.system.creatures)
        rabbit = self.system.spawn_creature("rabbit", 0, 0, energy=200.0)
        state = self.system.creature_ai(rabbit)
        assert state == "reproduce"
        offspring = [c for c in self.system.creatures
                     if c.species == "rabbit" and c is not rabbit]
        assert len(self.system.creatures) == before + 2
        assert len(offspring) == 1

    # ---- 食物链：完整层级 / 能量传递 / 捕食者受猎物限制 ----

    def test_case_06(self):
        """食物链有完整层级：草->兔->狼 三个营养级"""
        chain = self.system.food_chain()
        assert chain["grass"] == 0
        assert chain["rabbit"] == 1
        assert chain["wolf"] == 2

    def test_case_07(self):
        """食物链有能量传递：猎物能量按效率转化为捕食者能量"""
        assert self.system.energy_transfer(100.0) == pytest.approx(10.0)
        assert 0 < self.system.ENERGY_TRANSFER_EFFICIENCY < 1

    def test_case_08(self):
        """食物链有平衡：捕食者数量受猎物数量限制"""
        no_prey = Ecosystem()
        no_prey.set_population("rabbit", 0.0)
        with_prey = Ecosystem()
        for _ in range(30):
            no_prey.update(1.0)
            with_prey.update(1.0)
        assert no_prey.get_population("wolf") < 20.0
        assert with_prey.get_population("wolf") > no_prey.get_population("wolf")

    # ---- 种群动态：出生率死亡率 / 环境容量 ----

    def test_case_09(self):
        """种群动态：数量随出生率死亡率自然变化"""
        before = self.system.get_population("rabbit")
        self.system.population_dynamics("rabbit", dt=1.0)
        after = self.system.get_population("rabbit")
        assert after != before
        assert after > before  # 默认出生率 > 死亡率，种群增长

    def test_case_10(self):
        """种群动态有环境容量：数量不会无限增长"""
        capacity = 200.0
        self.system.set_population("rabbit", capacity)
        for _ in range(100):
            pop = self.system.population_dynamics("rabbit", dt=1.0)
            assert pop <= capacity + 1e-6
        self.system.set_population("rabbit", capacity * 0.95)
        for _ in range(200):
            pop = self.system.population_dynamics("rabbit", dt=1.0)
        assert pop <= capacity + 1e-6

    # ---- 迁徙：触发条件 / 方向 ----

    def test_case_11(self):
        """迁徙有触发条件：冬季触发迁徙，夏季不迁徙"""
        winter = self.system.migration("rabbit", season="winter")
        summer = self.system.migration("rabbit", season="summer")
        assert winter is not None
        assert summer is None
        # 非迁徙物种（草）任何季节都不迁徙
        assert self.system.migration("grass", season="winter") is None

    def test_case_12(self):
        """迁徙有方向：朝食物最丰富的栖息地迁徙"""
        result = self.system.migration("rabbit", season="winter")
        assert result["target"] == "south"  # 默认栖息地中 south 食物最丰富
        assert result["direction"] == (0.0, -1.0)

    # ---- 生态平衡：监控 / 灭绝级联 / 恢复机制 ----

    def test_case_13(self):
        """生态平衡有监控：报告健康度与各物种状态"""
        report = self.system.ecosystem_balance()
        assert 0.0 <= report["health"] <= 1.0
        assert report["status"] in ("healthy", "degraded", "collapsed")
        assert set(report["species"]) == {"grass", "rabbit", "wolf"}
        self.system.set_population("wolf", 0.0)
        report = self.system.ecosystem_balance()
        assert "wolf" in report["extinct"]
        assert report["species"]["wolf"]["status"] == "extinct"

    def test_case_14(self):
        """生态平衡有级联：一个物种灭绝会影响其他物种"""
        before = self.system.get_population("rabbit")
        self.system.set_population("grass", 0.0)
        for _ in range(10):
            self.system.update(1.0)
        assert self.system.get_population("rabbit") < before

    def test_case_15(self):
        """生态平衡有恢复机制：崩溃后可以恢复"""
        for species in ("grass", "rabbit", "wolf"):
            self.system.set_population(species, 0.0)
        collapsed = self.system.ecosystem_balance()
        assert collapsed["status"] == "collapsed"
        restored = self.system.recover()
        assert set(restored) == {"grass", "rabbit", "wolf"}
        recovered = self.system.ecosystem_balance()
        assert recovered["extinct"] == []
        assert recovered["health"] > collapsed["health"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
