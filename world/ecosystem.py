"""
游戏内生态系统与生物AI系统 - 核心模块

子系统：
- 生物AI：状态机（grow/flee/forage/reproduce/wander）+ 感知（食物/天敌/配偶）
- 食物链：草 -> 兔 -> 狼 完整层级，捕食有能量传递，捕食者受猎物数量限制
- 种群动态：出生率/死亡率 + 环境容量（逻辑斯蒂增长）
- 迁徙：有方向（季节目标点）+ 触发条件（秋冬季节）
- 生态平衡：灭绝级联、恢复机制、健康监控
"""
import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DEFAULT_SPECIES: Dict[str, Dict] = {
    "grass": {
        "trophic_level": 0, "eats": [], "max_energy": 30.0,
        "birth_rate": 0.30, "death_rate": 0.05, "carrying_capacity": 200,
        "speed": 0.0, "perception": 0.0, "repro_energy": 20.0,
    },
    "rabbit": {
        "trophic_level": 1, "eats": ["grass"], "max_energy": 60.0,
        "birth_rate": 0.25, "death_rate": 0.04, "carrying_capacity": 60,
        "speed": 3.0, "perception": 15.0, "repro_energy": 45.0,
    },
    "wolf": {
        "trophic_level": 2, "eats": ["rabbit"], "max_energy": 100.0,
        "birth_rate": 0.08, "death_rate": 0.06, "carrying_capacity": 15,
        "speed": 4.0, "perception": 20.0, "repro_energy": 80.0,
    },
}


@dataclass
class Creature:
    id: int
    species: str
    x: float
    y: float
    energy: float
    age: float = 0.0
    state: str = "idle"
    alive: bool = True


class Ecosystem:
    SEASONS = ("spring", "summer", "autumn", "winter")
    MIGRATION_SEASONS = ("autumn", "winter")
    EAT_RANGE = 1.5
    PREY_DEMAND_PER_PREDATOR = 5

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.width = float(self.config.get("width", 100.0))
        self.height = float(self.config.get("height", 100.0))
        self.energy_transfer_efficiency = float(
            self.config.get("energy_transfer_efficiency", 0.2))
        self.season_length = float(self.config.get("season_length", 50.0))
        self.rng = random.Random(self.config.get("seed", 42))
        self.species = copy.deepcopy(DEFAULT_SPECIES)
        for name, overrides in self.config.get("species", {}).items():
            if name in self.species:
                self.species[name].update(overrides)
        self.creatures: List[Creature] = []
        self.time = 0.0
        self._next_id = 0
        self._history: List[Dict] = []
        self._migration_events: List[Dict] = []
        self._births = {name: 0 for name in self.species}
        self._deaths = {name: 0 for name in self.species}
        self._birth_acc = {name: 0.0 for name in self.species}
        self._death_acc = {name: 0.0 for name in self.species}
        self.last_report: Optional[Dict] = None

    # ---------- 基础 ----------

    @property
    def season(self) -> str:
        idx = int(self.time // self.season_length) % len(self.SEASONS)
        return self.SEASONS[idx]

    def add_creature(self, species: str, x: float, y: float,
                     energy: Optional[float] = None) -> Creature:
        if species not in self.species:
            raise ValueError(f"未知物种: {species}")
        spec = self.species[species]
        creature = Creature(
            id=self._next_id, species=species, x=float(x), y=float(y),
            energy=float(energy) if energy is not None else float(spec["max_energy"]),
        )
        self._next_id += 1
        self.creatures.append(creature)
        return creature

    def population(self, species: Optional[str] = None) -> int:
        return sum(1 for c in self.creatures
                   if c.alive and (species is None or c.species == species))

    def _distance(self, a: Creature, b: Creature) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _purge_dead(self):
        self.creatures = [c for c in self.creatures if c.alive]

    def _spawn(self, species: str, count: int,
               near: Optional[Creature] = None) -> List[Creature]:
        made = []
        for _ in range(count):
            if near is not None:
                x = min(max(near.x + self.rng.uniform(-2.0, 2.0), 0.0), self.width)
                y = min(max(near.y + self.rng.uniform(-2.0, 2.0), 0.0), self.height)
            else:
                x = self.rng.uniform(0.0, self.width)
                y = self.rng.uniform(0.0, self.height)
            made.append(self.add_creature(
                species, x, y, energy=self.species[species]["max_energy"] * 0.5))
        return made

    def _kill(self, species: str, count: int):
        victims = [c for c in self.creatures if c.alive and c.species == species]
        victims.sort(key=lambda c: c.energy)  # 最弱的先死
        for c in victims[:count]:
            c.alive = False
            self._deaths[species] += 1
        self._purge_dead()

    def _clamp(self, creature: Creature):
        creature.x = min(max(creature.x, 0.0), self.width)
        creature.y = min(max(creature.y, 0.0), self.height)

    def _move_toward(self, creature: Creature, tx: float, ty: float, dist: float):
        dx, dy = tx - creature.x, ty - creature.y
        d = math.hypot(dx, dy)
        if d <= 0 or dist <= 0:
            return
        step = min(dist, d)
        creature.x += dx / d * step
        creature.y += dy / d * step
        self._clamp(creature)

    def _move_away(self, creature: Creature, fx: float, fy: float, dist: float):
        dx, dy = creature.x - fx, creature.y - fy
        d = math.hypot(dx, dy)
        if d <= 0:
            dx, dy, d = 1.0, 0.0, 1.0
        creature.x += dx / d * dist
        creature.y += dy / d * dist
        self._clamp(creature)

    # ---------- 生物AI：感知 + 状态机 ----------

    def perceive(self, creature: Creature) -> Dict[str, List[Creature]]:
        """感知：看到感知半径内的食物、天敌和配偶。"""
        spec = self.species[creature.species]
        radius = spec["perception"]
        sensed: Dict[str, List[Creature]] = {"food": [], "predators": [], "mates": []}
        for other in self.creatures:
            if other is creature or not other.alive:
                continue
            if self._distance(creature, other) > radius:
                continue
            other_spec = self.species[other.species]
            if other.species in spec["eats"]:
                sensed["food"].append(other)
            if creature.species in other_spec["eats"]:
                sensed["predators"].append(other)
            if other.species == creature.species:
                sensed["mates"].append(other)
        return sensed

    def creature_ai(self, creature: Creature, dt: float = 1.0) -> str:
        """状态机：flee(躲避) > forage(觅食) > reproduce(繁殖) > wander(游荡)。"""
        spec = self.species[creature.species]
        if spec["trophic_level"] == 0:
            creature.state = "grow"
            creature.energy = min(spec["max_energy"], creature.energy + 1.0 * dt)
            return creature.state

        sensed = self.perceive(creature)
        step = spec["speed"] * dt

        if sensed["predators"]:
            creature.state = "flee"
            threat = min(sensed["predators"],
                         key=lambda p: self._distance(creature, p))
            self._move_away(creature, threat.x, threat.y, step)
        elif creature.energy < 0.6 * spec["max_energy"] and sensed["food"]:
            creature.state = "forage"
            target = min(sensed["food"], key=lambda f: self._distance(creature, f))
            if self._distance(creature, target) <= self.EAT_RANGE:
                self.eat(creature, target)
            else:
                self._move_toward(creature, target.x, target.y, step)
        elif (creature.energy >= spec["repro_energy"] and sensed["mates"]
              and self.population(creature.species) < spec["carrying_capacity"]):
            creature.state = "reproduce"
            creature.energy *= 0.5
            self._spawn(creature.species, 1, near=creature)
        else:
            creature.state = "wander"
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            self._move_toward(creature,
                              creature.x + math.cos(angle) * step,
                              creature.y + math.sin(angle) * step, step)
        return creature.state

    # ---------- 食物链 ----------

    def food_chain(self) -> Dict:
        """完整食物链：营养层级 + 捕食关系 + 能量传递效率。"""
        levels: Dict[int, List[str]] = {}
        links: List[Tuple[str, str]] = []
        for name, spec in self.species.items():
            levels.setdefault(spec["trophic_level"], []).append(name)
            for prey in spec["eats"]:
                links.append((name, prey))
        return {"levels": levels, "links": links,
                "efficiency": self.energy_transfer_efficiency}

    def eat(self, predator: Creature, prey: Creature) -> float:
        """能量传递：捕食者获得 猎物能量 * 传递效率。"""
        spec = self.species[predator.species]
        gain = prey.energy * self.energy_transfer_efficiency
        predator.energy = min(spec["max_energy"], predator.energy + gain)
        prey.alive = False
        return gain

    # ---------- 种群动态 ----------

    def population_dynamics(self, dt: float = 1.0) -> Dict[str, Dict]:
        """逻辑斯蒂增长：出生率/死亡率驱动，受环境容量与猎物供给限制。"""
        changes: Dict[str, Dict] = {}
        for name, spec in self.species.items():
            n = self.population(name)
            if n <= 0:
                changes[name] = {"births": 0, "deaths": 0, "count": 0}
                continue
            capacity = spec["carrying_capacity"]
            prey_factor = 1.0
            if spec["eats"]:
                prey_total = sum(self.population(p) for p in spec["eats"])
                demand = max(1, n * self.PREY_DEMAND_PER_PREDATOR)
                prey_factor = min(1.0, prey_total / demand)
            self._birth_acc[name] += (spec["birth_rate"] * prey_factor * n
                                      * max(0.0, 1.0 - n / capacity) * dt)
            self._death_acc[name] += spec["death_rate"] * n * dt
            if spec["eats"] and prey_factor < 1.0:
                self._death_acc[name] += (1.0 - prey_factor) * 0.2 * n * dt
            births = int(self._birth_acc[name])
            self._birth_acc[name] -= births
            deaths = min(n, int(self._death_acc[name]))
            self._death_acc[name] -= int(self._death_acc[name])
            if births:
                self._spawn(name, births)
                self._births[name] += births
            if deaths:
                self._kill(name, deaths)
            changes[name] = {"births": births, "deaths": deaths,
                             "count": self.population(name)}
        return changes

    # ---------- 迁徙 ----------

    def migration_target(self) -> Tuple[float, float]:
        """迁徙方向：秋冬季节迁往温暖的南方低地。"""
        return (self.width / 2.0, self.height * 0.1)

    def migration(self, dt: float = 1.0) -> List[Dict]:
        """触发条件：进入秋冬迁徙季；动物朝迁徙目标点移动。"""
        events: List[Dict] = []
        if self.season not in self.MIGRATION_SEASONS:
            return events
        tx, ty = self.migration_target()
        for creature in self.creatures:
            if not creature.alive:
                continue
            spec = self.species[creature.species]
            if spec["trophic_level"] == 0:
                continue
            if math.hypot(tx - creature.x, ty - creature.y) < 1.0:
                continue
            old = (creature.x, creature.y)
            self._move_toward(creature, tx, ty, spec["speed"] * 2.0 * dt)
            events.append({"id": creature.id, "species": creature.species,
                           "from": old, "to": (creature.x, creature.y),
                           "reason": "season:" + self.season})
        self._migration_events.extend(events)
        return events

    # ---------- 生态平衡 ----------

    def ecosystem_balance(self, dt: float = 1.0) -> Dict:
        """健康监控 + 恢复机制；猎物灭绝通过种群动态级联影响捕食者。"""
        report: Dict = {"healthy": True, "species": {}, "issues": []}
        for name, spec in self.species.items():
            n = self.population(name)
            capacity = spec["carrying_capacity"]
            if n == 0:
                status = "extinct"
                report["healthy"] = False
                report["issues"].append(f"{name} 已灭绝")
            elif n < 0.1 * capacity:
                status = "critical"
                report["healthy"] = False
                report["issues"].append(f"{name} 数量危急")
                self._recover(name, dt)
            elif n < 0.25 * capacity:
                status = "recovering"
                self._recover(name, dt)
            else:
                status = "stable"
            report["species"][name] = {
                "count": n, "carrying_capacity": capacity, "status": status}
        self.last_report = report
        return report

    def _recover(self, species: str, dt: float):
        """恢复机制：数量低于阈值时提高有效出生率，帮助种群恢复。"""
        n = self.population(species)
        if n <= 0:
            return  # 已灭绝的物种无法自然恢复
        spec = self.species[species]
        bonus = max(1, int(spec["birth_rate"] * 2.0 * n * dt))
        self._spawn(species, bonus)
        self._births[species] += bonus

    # ---------- 主循环 ----------

    def update(self, dt: float):
        self.time += dt
        for creature in list(self.creatures):
            if creature.alive and self.species[creature.species]["trophic_level"] > 0:
                self.creature_ai(creature, dt)
        self._purge_dead()
        self.population_dynamics(dt)
        self.migration(dt)
        report = self.ecosystem_balance(dt)
        self._history.append({
            "time": self.time,
            "season": self.season,
            "populations": {name: self.population(name) for name in self.species},
        })
        return report

    def reset(self):
        self.creatures = []
        self.time = 0.0
        self._next_id = 0
        self._history = []
        self._migration_events = []
        for name in self.species:
            self._births[name] = 0
            self._deaths[name] = 0
            self._birth_acc[name] = 0.0
            self._death_acc[name] = 0.0
        self.last_report = None
