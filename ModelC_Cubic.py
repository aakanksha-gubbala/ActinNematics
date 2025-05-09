import os

import h5py
import numpy as np


class ModelC_2D:
    def __init__(self, seed, path, device='gpu'):
        self.device = device
        if self.device == 'gpu':
            import cupy as cp
            from cupy.fft import fft2, ifft2, fftfreq
            self.xp = cp
            self.fft2 = fft2
            self.ifft2 = ifft2
            self.fftfreq = fftfreq
        else:
            import numpy as np
            from numpy.fft import fft2, ifft2, fftfreq
            self.xp = np
            self.fft2 = fft2
            self.ifft2 = ifft2
            self.fftfreq = fftfreq

        self.K = None
        self.KX = None
        self.KY = None
        self.ALIASING = None

        self.N = 128
        self.L = 128
        self.DX = None
        self.DT = 1e-8
        self.TIME = 1000
        self.TOL = 1e-4

        self.prng = self.xp.random.RandomState(seed)
        self.PATH = path
        self.SKIPS = 10

        self.PHI_AVG = 0.3
        self.BETA = 1.0
        self.GAMMA = 1.0
        self.E1 = 1.0
        self.E3 = 1.0
        self.CHI = 1.0

    def grids(self):
        self.DX = self.L / (self.N - 1)
        kx = self.fftfreq(self.N, self.L / self.N) * 2 * self.xp.pi
        ky = self.fftfreq(self.N, self.L / self.N) * 2 * self.xp.pi
        self.KX, self.KY = self.xp.meshgrid(kx, ky)
        self.K = self.xp.sqrt(self.KX ** 2 + self.KY ** 2)
        self.K[self.K == 0] = 1 / self.N ** 2

        anti_alias_trunc = self.xp.max(self.xp.array([kx, ky])) * 2 / 3
        anti_alias = self.xp.array([self.xp.abs(kx) < anti_alias_trunc])
        if anti_alias.shape[0] == 1:
            anti_alias = anti_alias[0]
        anti_alias_mesh_x, anti_alias_mesh_y = self.xp.meshgrid(
            anti_alias, anti_alias)
        self.ALIASING = anti_alias_mesh_x * anti_alias_mesh_y

    def grad(self, v_fft):
        return (self.ifft2(1j * self.KX * v_fft).real,
                self.ifft2(1j * self.KY * v_fft).real)

    def curl_fft(self, vx, vy):
        return 1j * self.KX * self.fft2(vy) - 1j * self.KY * self.fft2(vx)

    def cubic(self, qxx_, qxx_fft, qxy_, qxy_fft):
        qxx_x, qxx_y = self.grad(qxx_fft)
        qxy_x, qxy_y = self.grad(qxy_fft)

        Qx = self.xp.array([[qxx_x, qxy_x],
                            [qxy_x, -qxx_x]])
        Qy = self.xp.array([[qxx_y, qxy_y],
                            [qxy_y, -qxx_y]])
        grad_Q = self.xp.array([Qx, Qy])

        Q = self.xp.array([[qxx_, qxy_],
                           [qxy_, -qxx_]])
        K = self.xp.array([self.KX, self.KY])
        F = self.xp.zeros(Q.shape)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        F[i][j] += (grad_Q[i][k][l] * grad_Q[j][k][l] -
                                    2 * grad_Q[l][k][l] * grad_Q[k][i][j] -
                                    2 * Q[k][l] * self.ifft2(-K[k] * K[l] * Q[i][j]).real)

        assert self.xp.all(F[0][1] == F[1][0])  # symmetry

        ones = self.xp.ones(qxx_.shape) * (F[0][0] + F[1][1])
        zeros = self.xp.zeros(qxx_.shape)
        I = self.xp.array([[ones, zeros],
                           [zeros, ones]])
        F = F - (I / 2)
        trF = F[0][0] + F[1][1]
        assert self.xp.sum(self.xp.abs(trF)) < self.TOL  # traceless
        return F[0][0], F[0][1]

    def euler_step(self, phi_, qxx_, qxy_, dt):
        phi_fft = self.fft2(phi_)
        qxx_fft = self.fft2(qxx_)
        qxy_fft = self.fft2(qxy_)

        phi_x, phi_y = self.grad(phi_fft)
        phi_grad_sq = phi_x ** 2 + phi_y ** 2

        TrQ2 = 2 * (qxx_ ** 2 + qxy_ ** 2)
        bulk_q = (1 + phi_) - self.BETA * TrQ2
        kappa_anch = 1 + np.abs(self.CHI) / 2

        anch_phi_fft = (1j * self.KX * self.fft2(qxx_ * phi_x + qxy_ * phi_y) +
                        1j * self.KY * self.fft2(qxy_ * phi_x - qxx_ * phi_y)) * self.ALIASING
        anch_qxx_fft = self.fft2(
            phi_x ** 2 - 0.5 * phi_grad_sq) * self.ALIASING
        anch_qxy_fft = self.fft2(phi_x * phi_y) * self.ALIASING

        fxx, fxy = self.cubic(qxx_, qxx_fft, qxy_, qxy_fft)

        bulk_qxx_fft = self.fft2(bulk_q * qxx_) * self.ALIASING
        fxx_fft = self.fft2(fxx) * self.ALIASING
        numerator_qxx = qxx_fft + dt * (bulk_qxx_fft - 0.5 * self.CHI * anch_qxx_fft +
                                        0.5 * self.E3 * fxx_fft)
        denominator_qxx = 1 + (self.E1 * self.K ** 2) * dt
        qxx_fft_1 = numerator_qxx / denominator_qxx

        bulk_qxy_fft = self.fft2(bulk_q * qxy_) * self.ALIASING
        fxy_fft = self.fft2(fxy) * self.ALIASING
        numerator_qxy = qxy_fft + dt * (bulk_qxy_fft - 0.5 * self.CHI * anch_qxy_fft +
                                        0.5 * self.E3 * fxy_fft)
        denominator_qxy = 1 + (self.E1 * self.K ** 2) * dt
        qxy_fft_1 = numerator_qxy / denominator_qxy

        f_eff_fft = self.fft2(phi_ ** 3 - phi_ - 0.5 * TrQ2) * self.ALIASING
        numerator_phi = (phi_fft * (1 + self.K ** 6 * dt) -
                         self.GAMMA * self.K ** 2 * dt * (f_eff_fft - self.CHI * anch_phi_fft))
        denominator_phi = 1 + (kappa_anch * self.GAMMA *
                               self.K ** 4 + self.K ** 6) * dt
        phi_fft_1 = numerator_phi / denominator_phi
        return self.ifft2(phi_fft_1).real, self.ifft2(qxx_fft_1).real, self.ifft2(qxy_fft_1).real

    def run(self, phi_init=None, Qxx_init=None, Qxy_init=None):
        self.grids()
        noise = self.prng.normal(loc=0, scale=1e-2, size=(self.N, self.N))

        if phi_init is None:
            phi = noise + self.PHI_AVG
            Qxx = noise
            Qxy = noise
        else:
            if self.device == 'gpu':
                phi = self.xp.array(phi_init)
                Qxx = self.xp.array(Qxx_init)
                Qxy = self.xp.array(Qxy_init)
            else:
                phi = phi_init
                Qxx = Qxx_init
                Qxy = Qxy_init

        T = 0
        counter = 0

        delta = 1

        time = []
        frames_phi = []
        frames_qxx = []
        frames_qxy = []
        r = 1e-3
        while T <= self.TIME:
            if self.xp.isnan(phi).any() or self.xp.isinf(phi).any():
                print("-----------------------------------------")
                print("Simulation blowup.")
                print("-----------------------------------------")
                break
            elif self.DT < 1e-12:
                print("-----------------------------------------")
                print("Time step is less than picoseconds.")
                print("-----------------------------------------")
                break
            else:
                if counter % self.SKIPS == 0 or np.abs(T - self.TIME) < self.DT:
                    time.append(float(T))
                    if self.device == 'gpu':
                        frames_qxx.append(self.xp.asnumpy(Qxx))
                        frames_qxy.append(self.xp.asnumpy(Qxy))
                        frames_phi.append(self.xp.asnumpy(phi))
                    else:
                        frames_qxx.append(Qxx)
                        frames_qxy.append(Qxy)
                        frames_phi.append(phi)

                    print("--------------------- %d ---------------------" % counter)
                    print("Time step  = %.2e" % self.DT)
                    print("Time       = %.2f" % T)
                    print("avg(phi)   = %.2f" % self.xp.average(phi))
                    print("min(phi)   = %.2f | max(phi)   = %.2f" %
                          (phi.min(), phi.max()))
                    print("min(Qxx)   = %.2f | max(Qxx)   = %.2f" %
                          (Qxx.min(), Qxx.max()))
                    print("min(Qxy)   = %.2f | max(Qxy)   = %.2f" %
                          (Qxy.min(), Qxy.max()))
                    print("Delta      = %.1e" % delta)
            (phi_half_1, Qxx_half_1, Qxy_half_1) = self.euler_step(phi,
                                                                   Qxx, Qxy,
                                                                   self.DT / 2)
            (phi_half_2, Qxx_half_2, Qxy_half_2) = self.euler_step(phi_half_1,
                                                                   Qxx_half_1, Qxy_half_1,
                                                                   self.DT / 2)
            (phi, Qxx, Qxy) = self.euler_step(phi,
                                              Qxx, Qxy,
                                              self.DT)

            delta_phi = self.xp.average(self.xp.abs(phi - phi_half_2))
            delta_qxx = self.xp.average(self.xp.abs(Qxx - Qxx_half_2))
            delta_qxy = self.xp.average(self.xp.abs(Qxy - Qxy_half_2))
            delta = self.xp.power(delta_phi * delta_qxx * delta_qxy, 1 / 3)
            if delta != 0:
                self.DT = self.DT * self.xp.power(self.TOL / delta, 1 / 5)
            T += self.DT
            counter += 1
            del phi_half_1, Qxx_half_1, Qxy_half_1
            del phi_half_2, Qxx_half_2, Qxy_half_2
            if self.device == 'gpu':
                self.xp.get_default_memory_pool().free_all_blocks()
        print("----------------- Done! -----------------")
        for var, name in zip([time,
                              frames_phi,
                              frames_qxx,
                              frames_qxy],
                             ['time',
                              'phi',
                              'qxx',
                              'qxy']):
            h5_file = h5py.File(self.PATH + '/' + name + '.h5', 'w')
            h5_file.create_dataset(name, data=var)
            h5_file.close()
        del self.prng

    def export_params(self):
        with open(os.path.join(self.PATH, 'params.txt'), 'w') as f:
            f.write(f"path        {self.PATH}\n")
            f.write(f"N           {self.N}\n")
            f.write(f"L           {self.L}\n")
            f.write(f"DX          {self.DX}\n")
            f.write(f"TIME        {self.TIME}\n")
            f.write(f"TOL         {self.TOL}\n")
            f.write(f"PHI_AVG     {self.PHI_AVG}\n")
            f.write(f"BETA        {self.BETA}\n")
            f.write(f"GAMMA       {self.GAMMA}\n")
            f.write(f"E1          {self.E1}\n")
            f.write(f"E3          {self.E3}\n")
            f.write(f"CHI         {self.CHI}\n")
            f.write(f"device      {self.device}\n")
