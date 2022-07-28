class EnvTest(object):
    def __init__(self):
        self.scale = 1
        self.state = 1 * self.scale
        self.count = 0
        self.state_number = 100

    def reset(self):
        self.state = 1 * self.scale
        self.count = 0

    def step(self, action):
        assert action in [0, 1], 'action limit!'
        done = bool(self.state >= self.state_number * self.scale)
        self.state += self.scale
        self.count += action
        reward = action if not done else action -self.count * 2
        # reward = -10
        return self.state, reward, done

