"""
游戏内生态系统与生物AI系统 - 核心模块

功能：
- 生物AI：状态机（idle/forage/flee/reproduce/migrate）+ 感知（发现食物与天敌）
- 食物链：草 -> 兔 -> 狼 完整层级，能量传递，捕食者数量受猎物限制
- 种群动态：出生率/死亡率 + 环境容量（逻辑斯蒂增长）
- 迁徙：季节/食物短缺触发，朝食物最丰富的栖息地迁徙
- 生态平衡：健康监控 + 灭绝级联 + 恢复机制
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STATE_IDLE = "idle"
STATE_FORAGE = "forage"
STATE_FLEE = "flee"
STATE_REPRODUCE = "reproduce"
STATE_MIGRATE = "migrate"
AI_STATES = (STATE_IDLE, STATE_FORAGE, STATE_FLEE, STATE_REPRODUCE, STATE_MIGRATE)

SEASONS = ("spring", "summer", "autumn", "winter")

# 各季节对生产者（草）生长速率的影响
SEASON_GROWTH_FACTOR = {
    "spring": 1.2,
    "summer": 1.0,
    "autumn": 0.8,
    "winter": 0.5,
}

# 栖息地及其方向向量（单位向量）
HABITAT_DIRECTIONS = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}

DEFAULT_HABITATS = {
    "north": 0.3,
    "south": 0.9,
    "east": 0.5,
    "west": 0.4,
}

DEFAULT_SPECIES = {
    "grass": {
        "birth_rate": 0.5,
        "death_rate": 0.05,
        "carrying_capacity": 1000.0,
        "initial": 800.0,
        "energy": 10.0,
        "diet": [],
        "perception": 0.0,
        "speed": 0.0,
        "migratory": False,
    },
    "rabbit": {
        "birth_rate": 0.3,
        "death_rate": 0.1,
        "carrying_capacity": 200.0,
        "initial": 100.0,
        "energy": 50.0,
        "diet": ["grass"],
        "perception": 10.0,
        "speed": 2.0,
        "migratory": True,
    },
    "wolf": {
        "birth_rate": 0.25,
        "death_rate": 0.15,
        "carrying_capacity": 40.0,
        "initial": 20.0,
        "energy": 100.0,
        "diet": ["rabbit"],
        "perception": 15.0,
        "speed": 3.0,
        "migratory": True,
    },
}

# 半饱和常数：猎物数量达到该值时，捕食者获得一半的食物可获得性
HALF_SATURATION = 20.0
# 个体AI：能量低于该比例 * 基础能量时感到饥饿
HUNGER_RATIO = 0.5
# 个体AI：能量高于该比例 * 基础能量时可以繁殖
REPRODUCE_RATIO = 1.5
# 个体AI：吃到食物的最大距离
EAT_DISTANCE = 1.0
# 最小可存活种群（占环境容量比例），低于该值视为功能性灭绝
MIN_VIABLE_RATIO = 0.05


@dataclass
class Creature:
    """生态系统中的一个个体生物。"""

    species: str
    x: float = 0.0
    y: float = 0.0
    energy: float = 0.0
    state: str = STATE_IDLE

    def distance_to(self, other: "Creature") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Ecosystem:
    """游戏内生态系统与生物AI系统。"""

    # 营养级之间的能量传递效率
    ENERGY_TRANSFER_EFFICIENCY = 0.1

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._rng = random.Random(self.config.get("seed", 42))
        self._season_length = float(self.config.get("season_length", 100.0))
        self._habitats = dict(self.config.get("habitats", DEFAULT_HABITATS))

        self._species: Dict[str, Dict] = {}
        overrides = self.config.get("species", {})
        for name, params in DEFAULT_SPECIES.items():
            merged = dict(params)
            merged.update(overrides.get(name, {}))
            self._species[name] = merged

        self._populations: Dict[str, float] = {}
        self._creatures: List[Creature] = []
        self._time = 0.0
        self._history: List[Dict] = []
        self.reset()

    # ------------------------------------------------------------------
    # 基础接口
    # ------------------------------------------------------------------

    def reset(self):
        self._populations = {
            name: float(params["initial"])
            for name, params in self._species.items()
        }
        self._creatures = []
        self._time = 0.0
        self._history = []

    @property
    def season(self) -> str:
        index = int(self._time / self._season_length) % len(SEASONS)
        return SEASONS[index]

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    def get_population(self, species: str) -> float:
        return self._populations.get(species, 0.0)

    def set_population(self, species: str, value: float):
        self._populations[species] = max(0.0, float(value))

    def spawn_creature(self, species: str, x: float = 0.0, y: float = 0.0,
                       energy: Optional[float] = None) -> Creature:
        params = self._species[species]
        creature = Creature(
            species=species,
            x=float(x),
            y=float(y),
            energy=float(energy if energy is not None else params["energy"]),
        )
        self._creatures.append(creature)
        return creature

    @property
    def creatures(self) -> List[Creature]:
        return list(self._creatures)

    # ------------------------------------------------------------------
    # 生物AI：感知 + 状态机
    # ------------------------------------------------------------------

    def _predators_of(self, species: str) -> List[str]:
        return [
            name for name, params in self._species.items()
            if species in params.get("diet", [])
        ]

    def _perceive(self, creature: Creature, species_names: List[str]) -> List[Creature]:
        """感知：返回感知半径内、属于给定物种的个体，按距离升序。"""
        perception = self._species[creature.species].get("perception", 0.0)
        if perception <= 0 or not species_names:
            return []
        seen = [
            other for other in self._creatures
            if other is not creature
            and other.species in species_names
            and creature.distance_to(other) <= perception
        ]
        seen.sort(key=creature.distance_to)
        return seen

    def creature_ai(self, creature: Creature, dt: float = 1.0) -> str:
        """生物AI状态机：感知环境并决定觅食、躲避、繁殖或迁徙。"""
        params = self._species[creature.species]
        base_energy = params["energy"]
        speed = params.get("speed", 0.0)

        threats = self._perceive(creature, self._predators_of(creature.species))
        food = self._perceive(creature, params.get("diet", []))

        if threats:
            # 躲避：朝远离最近天敌的方向移动
            creature.state = STATE_FLEE
            threat = threats[0]
            dx, dy = creature.x - threat.x, creature.y - threat.y
            dist = math.hypot(dx, dy) or 1.0
            creature.x += dx / dist * speed * dt
            creature.y += dy / dist * speed * dt
        elif creature.energy < base_energy * HUNGER_RATIO and food:
            # 觅食：朝最近的食物移动，足够近则吃掉并补充能量
            creature.state = STATE_FORAGE
            target = food[0]
            dist = creature.distance_to(target)
            if dist <= EAT_DISTANCE:
                creature.energy += target.energy
                self._creatures.remove(target)
            else:
                step = min(speed * dt, dist)
                creature.x += (target.x - creature.x) / dist * step
                creature.y += (target.y - creature.y) / dist * step
        elif creature.energy >= base_energy * REPRODUCE_RATIO:
            # 繁殖：消耗一半能量产生后代
            creature.state = STATE_REPRODUCE
            creature.energy *= 0.5
            self.spawn_creature(creature.species, creature.x, creature.y,
                                energy=base_energy)
        elif params.get("migratory") and self._migration_triggered(creature.species):
            creature.state = STATE_MIGRATE
            migration = self.migration(creature.species)
            if migration:
                dx, dy = migration["direction"]
                creature.x += dx * speed * dt
                creature.y += dy * speed * dt
        else:
            creature.state = STATE_IDLE

        return creature.state

    # ------------------------------------------------------------------
    # 食物链：层级 + 能量传递
    # ------------------------------------------------------------------

    def food_chain(self) -> Dict[str, int]:
        """返回每个物种所在的营养级（生产者=0，逐级递增）。"""
        levels: Dict[str, int] = {}

        def level_of(name: str, seen: Tuple[str, ...] = ()) -> int:
            if name in levels:
                return levels[name]
            diet = [p for p in self._species[name].get("diet", [])
                    if p in self._species and p not in seen]
            if not diet:
                levels[name] = 0
            else:
                levels[name] = 1 + max(level_of(p, seen + (name,)) for p in diet)
            return levels[name]

        for name in self._species:
            level_of(name)
        return levels

    def energy_transfer(self, prey_energy: float) -> float:
        """能量传递：猎物能量按传递效率转化为捕食者能量。"""
        return prey_energy * self.ENERGY_TRANSFER_EFFICIENCY

    def _food_availability(self, species: str) -> float:
        """食物可获得性（0~1）：捕食者能获得的能量受猎物数量限制。"""
        diet = self._species[species].get("diet", [])
        if not diet:
            return SEASON_GROWTH_FACTOR[self.season]
        availabilities = []
        for prey in diet:
            prey_pop = self._populations.get(prey, 0.0)
            availabilities.append(prey_pop / (prey_pop + HALF_SATURATION))
        return min(availabilities)

    def _predation(self, dt: float):
        """捕食：捕食者按可获得性消耗猎物，猎物减少、能量沿食物链传递。"""
        for predator, params in self._species.items():
            predator_pop = self._populations.get(predator, 0.0)
            if predator_pop <= 0:
                continue
            for prey in params.get("diet", []):
                prey_pop = self._populations.get(prey, 0.0)
                if prey_pop <= 0:
                    continue
                availability = prey_pop / (prey_pop + HALF_SATURATION)
                kills = 0.02 * predator_pop * availability * dt
                kills = min(kills, prey_pop)
                self._populations[prey] = prey_pop - kills

    # ------------------------------------------------------------------
    # 种群动态：出生率/死亡率 + 环境容量
    # ------------------------------------------------------------------

    def population_dynamics(self, species: str, dt: float) -> float:
        """逻辑斯蒂增长：dN = r_eff * N * (1 - N/K)，种群数量受环境容量限制。"""
        params = self._species[species]
        pop = self._populations.get(species, 0.0)
        capacity = params["carrying_capacity"]
        availability = self._food_availability(species)
        effective_rate = params["birth_rate"] * availability - params["death_rate"]

        growth = effective_rate * pop * (1.0 - pop / capacity) * dt
        new_pop = pop + growth
        if pop <= capacity:
            new_pop = min(new_pop, capacity)
        new_pop = max(0.0, new_pop)
        self._populations[species] = new_pop
        return new_pop

    # ------------------------------------------------------------------
    # 迁徙：方向 + 触发条件
    # ------------------------------------------------------------------

    def _migration_triggered(self, species: str) -> bool:
        params = self._species[species]
        if not params.get("migratory"):
            return False
        if self.season in params.get("migration_seasons", ("winter",)):
            return True
        # 食物短缺也触发迁徙
        diet = params.get("diet", [])
        for prey in diet:
            prey_capacity = self._species[prey]["carrying_capacity"]
            if self._populations.get(prey, 0.0) < prey_capacity * 0.1:
                return True
        return False

    def migration(self, species: str, season: Optional[str] = None) -> Optional[Dict]:
        """迁徙：由季节或食物短缺触发，方向为食物最丰富的栖息地。"""
        params = self._species[species]
        if not params.get("migratory"):
            return None

        season = season or self.season
        triggered = season in params.get("migration_seasons", ("winter",))
        reason = "season" if triggered else ""
        if not triggered:
            diet = params.get("diet", [])
            for prey in diet:
                prey_capacity = self._species[prey]["carrying_capacity"]
                if self._populations.get(prey, 0.0) < prey_capacity * 0.1:
                    triggered = True
                    reason = "food_scarcity"
                    break
        if not triggered:
            return None

        # 方向：朝食物最丰富的栖息地
        target = max(self._habitats, key=lambda h: self._habitats[h])
        direction = HABITAT_DIRECTIONS[target]
        return {
            "species": species,
            "direction": direction,
            "target": target,
            "season": season,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # 生态平衡：监控 + 恢复机制
    # ------------------------------------------------------------------

    def ecosystem_balance(self) -> Dict:
        """监控生态健康：各物种状态、灭绝名单与整体健康度（0~1）。"""
        species_report = {}
        extinct = []
        health_sum = 0.0
        for name, params in self._species.items():
            capacity = params["carrying_capacity"]
            pop = self._populations.get(name, 0.0)
            ratio = pop / capacity if capacity > 0 else 0.0
            if pop < capacity * MIN_VIABLE_RATIO:
                status = "extinct"
                extinct.append(name)
            elif ratio < 0.25:
                status = "endangered"
            else:
                status = "stable"
            species_report[name] = {
                "population": pop,
                "capacity": capacity,
                "ratio": ratio,
                "status": status,
            }
            health_sum += min(1.0, ratio / 0.5)

        health = health_sum / len(self._species) if self._species else 0.0
        if health >= 0.8:
            overall = "healthy"
        elif health >= 0.4:
            overall = "degraded"
        else:
            overall = "collapsed"
        return {
            "health": health,
            "status": overall,
            "species": species_report,
            "extinct": extinct,
        }

    def recover(self):
        """恢复机制：把功能性灭绝的物种重新引入到最小可存活种群之上。"""
        restored = []
        for name, params in self._species.items():
            capacity = params["carrying_capacity"]
            pop = self._populations.get(name, 0.0)
            if pop < capacity * MIN_VIABLE_RATIO:
                self._populations[name] = capacity * 0.2
                restored.append(name)
        return restored

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def update(self, dt: float):
        self._time += dt
        for name in self._species:
            self.population_dynamics(name, dt)
        self._predation(dt)
        for creature in list(self._creatures):
            self.creature_ai(creature, dt)
        self._history.append({
            "time": self._time,
            "season": self.season,
            "populations": dict(self._populations),
        })
