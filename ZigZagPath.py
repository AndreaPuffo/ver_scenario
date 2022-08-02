import numpy as np


class ZigZagPath():
    """
    creates an environment with an agent, walls, target set
    the environment
    __________________
    |           |    |
    |   |       |    |
    |   |       |    |
    |   |       |    |
    |   |       |    |
    |___| ___________|

    target set in top right corner
    state space is 10 x 10 grid

    """

    def __init__(self, x0):
        self.bounds = [10, 10]
        self.current_state = x0
        self.walls = self.create_walls()
        self.target = np.array([
            [8., 8., 8., 9., 9., 9.],
            [7., 8., 9., 7., 8., 9.]
        ])

    def create_walls(self):
        """
        cretaes the obstacles / walls
        :return: array, obstacles coordinates
        """

        # first_wall
        x1 = np.repeat(3., 6 + 1)
        y1 = np.arange(start=0, stop=6 + 1, step=1)
        first_wall = np.stack([x1, y1])

        x2 = np.repeat(7., 6 + 1)
        y2 = np.arange(start=3, stop=9 + 1, step=1)
        second_wall = np.stack([x2, y2])

        walls = np.hstack([first_wall, second_wall])

        return walls

    def out_of_bounds(self, state):
        """
        checks if the state is out of bounds (i.e. the 10x10 grid)
        :param state: array
        :return: True if state is out of the state space
        """
        if state[0] < 0 or state[0] > 9 or state[1] < 0 or state[1] > 9:
            return True
        else:
            return False

    def into_walls(self, state):
        """
        checks if the state is crashed into the obstacles
        :param state: array
        :return: True if state crashed into the obstacles
        """
        if any([(state.reshape(self.walls[:, i].shape) == self.walls[:, i]).all() for i in range(self.walls.shape[1])]):
            return True
        else:
            return False

    def in_target(self, state):
        """
        checks if state is in target set
        :param state: array
        :return: True if state is in target set
        """
        if any([(state.reshape(self.target[:, i].shape) == self.target[:, i]).all()
                for i in range(self.target.shape[1])]):
            return True
        else:
            return False

    def step(self, action):
        """
        basic method of the environment. advances the agent by one step.
        :param action: str, indicates which action the agent should take
        :return:
            self.current_state: array, the updated state
            reward: float,
            done: bool, True if crashed or out of bounds
        """

        # check if started in walls
        reward = self.reward_state(self.current_state)
        if reward < 0:
            return self.current_state, reward, True

        if action == 'u':
            update = [0, 1]
        elif action == 'd':
            update = [0, -1]
        elif action == 'e':
            update = [1, 0]
        elif action == 'w':
            update = [-1, 0]
        else:
            raise ValueError('Not recognised Action')

        self.current_state = self.current_state + np.array([update]).T
        reward = self.reward_state(self.current_state)

        done = False
        # if crashed, out, we are done
        if self.into_walls(self.current_state) or \
                self.out_of_bounds(self.current_state):
            done = True

        return self.current_state, reward, done

    def reward_state(self, state):
        """
        computes state reward
        :param state: array
        :return: float, the reward corresponding to the state
        """

        reward = 0.

        if self.out_of_bounds(state):
            reward = -10.
        elif self.into_walls(state):
            reward = -10.
        elif self.in_target(state):
            reward = 10.

        return reward


# tests
def zzp_testing():
    # check if the walls are actually considered walls
    def check_walls():
        states = zzp.create_walls()

        for idx in range(states.shape[1]):
            s = states[:, idx]
            assert zzp.into_walls(s)
            assert not zzp.in_target(s)

    # check if target is actually considered target
    def check_target():
        states = zzp.target

        for idx in range(states.shape[1]):
            s = states[:, idx]
            assert not zzp.into_walls(s)
            assert zzp.in_target(s)

    zzp = ZigZagPath(x0=np.array([0., 0.]))
    check_walls()
    check_target()

    # test a random state if it is in the obstacle and not everywhere else
    test_state = np.array([[7.], [8.]])
    rew = zzp.reward_state(test_state)
    assert zzp.into_walls(test_state)
    assert rew == -1.
    assert not zzp.in_target(test_state)

    # test a random target state if is actually target and nothing else
    test_state = np.array([[8.], [9.]])
    rew = zzp.reward_state(test_state)
    assert not zzp.into_walls(test_state)
    assert rew == 10.
    assert zzp.in_target(test_state)


if __name__ == '__main__':
    zzp_testing()
