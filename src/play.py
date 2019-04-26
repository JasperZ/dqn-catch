from getkey import getkey, keys
from catch import Catch, Actions

env = Catch(5, 5, True, False)

state = env.getState()
done = False

print(state)
print()

while not done:
    key = getkey()

    if key == keys.LEFT:
        action = Actions.LEFT
    elif key == keys.RIGHT:
        action = Action.RIGHT
    elif key == keys.UP:
        action = Actions.UP
    elif key == keys.DOWN:
        action = Actions.DOWN

    reward, state, done = env.move(action)
    print(state)
    print("reward: {}".format(reward))
    print("done: {}".format(done))
    print()
