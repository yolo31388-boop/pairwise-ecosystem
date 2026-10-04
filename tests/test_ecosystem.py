"""
游戏内生态系统与生物AI系统 - 验收测试
运行方式：python -m pytest tests/test_ecosystem.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.ecosystem import Ecosystem


class TestEcosystem:
    def setup_method(self):
        self.system = Ecosystem({"seed": 7})

    # ---------- 生物AI ----------

    def test_case_01(self):
        """AI感知：能看到半径内的食物和天敌，看不到半径外的"""
        eco = self.system
        rabbit = eco.add_creature("rabbit", 10, 10)
        grass = eco.add_creature("grass", 15, 10)
        far_wolf = eco.add_creature("wolf", 90, 90)
        sensed = eco.perceive(rabbit)
        assert grass in sensed["food"]
        assert far_wolf not in sensed["predators"]
        near_wolf = eco.add_creature("wolf", 20, 10)
        mate = eco.add_creature("rabbit", 12, 10)
        sensed = eco.perceive(rabbit)
        assert near_wolf in sensed["predators"]
        assert mate in sensed["mates"]

    def test_case_02(self):
        """AI状态机：天敌靠近时进入flee状态并远离天敌"""
        eco = self.system
        rabbit = eco.add_creature("rabbit", 50, 50)
        wolf = eco.add_creature("wolf", 53, 50)
        dist_before = eco._distance(rabbit, wolf)
        state = eco.creature_ai(rabbit, 1.0)
        assert state == "flee"
        assert rabbit.state == "flee"
        assert eco._distance(rabbit, wolf) > dist_before

    def test_case_03(self):
        """AI觅食：饥饿时朝食物移动，靠近后进食并增加能量"""
        eco = self.system
        rabbit = eco.add_creature("rabbit", 50, 50, energy=10)
        grass = eco.add_creature("grass", 56, 50)
        dist_before = eco._distance(rabbit, grass)
        state = eco.creature_ai(rabbit, 1.0)
        assert state == "forage"
        assert eco._distance(rabbit, grass) < dist_before
        grass.x, grass.y = rabbit.x + 1.0, rabbit.y
        energy_before = rabbit.energy
        eco.creature_ai(rabbit, 1.0)
        assert rabbit.energy > energy_before
        assert not grass.alive

    def test_case_04(self):
        """AI繁殖：能量充足且有配偶时进入reproduce状态并产生后代"""
        eco = self.system
        rabbit = eco.add_creature("rabbit", 50, 50, energy=50)
        eco.add_creature("rabbit", 51, 50, energy=50)
        assert eco.population("rabbit") == 2
        state = eco.creature_ai(rabbit, 1.0)
        assert state == "reproduce"
        assert eco.population("rabbit") == 3

    # ---------- 食物链 ----------

    def test_case_05(self):
        """食物链有完整层级：草->兔->狼三个营养级"""
        chain = self.system.food_chain()
        assert len(chain["levels"]) == 3
        assert chain["levels"][0] == ["grass"]
        assert chain["levels"][1] == ["rabbit"]
        assert chain["levels"][2] == ["wolf"]
        assert ("rabbit", "grass") in chain["links"]
        assert ("wolf", "rabbit") in chain["links"]

    def test_case_06(self):
        """能量传递：吃掉猎物后捕食者能量增加"""
        eco = self.system
        wolf = eco.add_creature("wolf", 0, 0, energy=50)
        rabbit = eco.add_creature("rabbit", 0, 0, energy=50)
        gain = eco.eat(wolf, rabbit)
        assert gain == pytest.approx(50 * eco.energy_transfer_efficiency)
        assert wolf.energy == pytest.approx(60)
        assert not rabbit.alive

    def test_case_07(self):
        """食物链平衡：没有猎物时捕食者数量下降"""
        eco = self.system
        for _ in range(10):
            eco.add_creature("wolf", 50, 50)
        before = eco.population("wolf")
        for _ in range(5):
            eco.population_dynamics(1.0)
        assert eco.population("wolf") < before

    # ---------- 种群动态 ----------

    def test_case_08(self):
        """种群动态：数量自然变化，有出生"""
        eco = self.system
        for _ in range(150):
            eco.add_creature("grass", 50, 50)
        for _ in range(10):
            eco.add_creature("rabbit", 50, 50)
        before = eco.population("rabbit")
        for _ in range(10):
            eco.population_dynamics(1.0)
        assert eco.population("rabbit") != before
        assert eco._births["rabbit"] > 0

    def test_case_09(self):
        """种群动态：超过环境容量时死亡率主导，数量下降"""
        eco = self.system
        for _ in range(200):
            eco.add_creature("grass", 50, 50)
        for _ in range(100):
            eco.add_creature("rabbit", 50, 50)
        before = eco.population("rabbit")
        for _ in range(5):
            eco.population_dynamics(1.0)
        assert eco._deaths["rabbit"] > 0
        assert eco.population("rabbit") < before

    def test_case_10(self):
        """种群动态：环境容量限制，数量不会无限增长"""
        eco = self.system
        capacity = eco.species["rabbit"]["carrying_capacity"]
        for _ in range(150):
            eco.add_creature("grass", 50, 50)
        for _ in range(10):
            eco.add_creature("rabbit", 50, 50)
        peak = 0
        for _ in range(300):
            eco.population_dynamics(1.0)
            peak = max(peak, eco.population("rabbit"))
        assert peak <= capacity
        assert eco.population("rabbit") > 0

    # ---------- 迁徙 ----------

    def test_case_11(self):
        """迁徙：触发后动物朝迁徙目标点移动"""
        eco = self.system
        eco.time = 2 * eco.season_length  # 秋季
        rabbit = eco.add_creature("rabbit", 50, 80)
        tx, ty = eco.migration_target()
        dist_before = ((rabbit.x - tx) ** 2 + (rabbit.y - ty) ** 2) ** 0.5
        events = eco.migration(1.0)
        assert len(events) > 0
        dist_after = ((rabbit.x - tx) ** 2 + (rabbit.y - ty) ** 2) ** 0.5
        assert dist_after < dist_before

    def test_case_12(self):
        """迁徙触发条件：非迁徙季不迁徙，换季后才迁徙"""
        eco = self.system
        rabbit = eco.add_creature("rabbit", 50, 80)
        eco.time = 0.0  # 春季
        assert eco.migration(1.0) == []
        assert (rabbit.x, rabbit.y) == (50, 80)
        eco.time = 3 * eco.season_length  # 冬季
        assert len(eco.migration(1.0)) > 0
        assert (rabbit.x, rabbit.y) != (50, 80)

    # ---------- 生态平衡 ----------

    def test_case_13(self):
        """生态平衡：猎物灭绝会级联影响捕食者"""
        eco = self.system
        for _ in range(200):
            eco.add_creature("grass", 50, 50)
        for _ in range(60):
            eco.add_creature("rabbit", 50, 50)
        for _ in range(10):
            eco.add_creature("wolf", 50, 50)
        for _ in range(10):
            eco.population_dynamics(1.0)
        assert eco.population("wolf") > 0
        for c in eco.creatures:
            if c.species == "rabbit":
                c.alive = False
        eco._purge_dead()
        for _ in range(15):
            eco.population_dynamics(1.0)
        assert eco.population("wolf") == 0

    def test_case_14(self):
        """生态平衡：恢复机制让濒危种群恢复"""
        eco = self.system
        for _ in range(150):
            eco.add_creature("grass", 50, 50)
        for _ in range(3):
            eco.add_creature("rabbit", 50, 50)
        for _ in range(30):
            eco.population_dynamics(1.0)
            eco.ecosystem_balance(1.0)
        assert eco.population("rabbit") >= 10

    def test_case_15(self):
        """生态平衡：健康监控能报告生态是否健康"""
        eco = self.system
        for _ in range(100):
            eco.add_creature("grass", 50, 50)
        for _ in range(20):
            eco.add_creature("rabbit", 50, 50)
        for _ in range(5):
            eco.add_creature("wolf", 50, 50)
        report = eco.ecosystem_balance(1.0)
        assert report["healthy"] is True
        assert report["species"]["rabbit"]["status"] == "stable"
        for c in eco.creatures:
            if c.species == "rabbit":
                c.alive = False
        eco._purge_dead()
        report = eco.ecosystem_balance(1.0)
        assert report["healthy"] is False
        assert report["species"]["rabbit"]["status"] == "extinct"
        assert len(report["issues"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
