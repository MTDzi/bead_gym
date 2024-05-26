from beads_gym.environment.environment import Environment
import beads_gym.environment.environment as e
print(e.__file__)
import beads_gym.beads as b
print(b.__file__)
from beads_gym.beads.beads import Bead, ThreeDegreesOfFreedomBead


bead = Bead([1, 1, 1])
tdf_bead = ThreeDegreesOfFreedomBead([1, 1, 1], 1.0)

# env = Environment([bead, tdf_bead])
env = Environment()
env.add_bead(bead)
env.add_bead(tdf_bead)

print(env.get_beads())