from model.coop_pomdp_from import CoopPOMDP

class Tiger(CoopPOMDP):
    def __init__(self):
        super().__init__(2, 3, 2, 2)

    def _set_tro(self):
        self.t[:2, :, :, 1] = 1
        self.t[2, :, 0, 0] = 1
        self.t[2, :, 1, 1] = 1

        self.r[0, :, 0, :] = [10, -100]
        self.r[1, :, 0, :] = [-100, 10]
        self.r[2, :, 0, :] = -1

        self.o[0, :, :, 0] = 1
        self.o[0, 2, 0, :] = [0.8, 0.2]
        self.o[1, :, :, 1] = 1
        self.o[1, 2, 0, :] = [0.2, 0.8]
