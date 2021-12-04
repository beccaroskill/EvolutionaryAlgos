from vpython import scene, vector, mag, norm, proj, color, \
                    cylinder, box, sphere
from PIL import Image, ImageGrab
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy.random as random
import math


SHADOW_COLOR = vector(0.7, 0.7, 0.7)
SHADOW_HEIGHT = 10**(-5)

class Mass:
    
    G = vector(0, -9.81, 0)
    DEFAULT_DAMPEN = 1
    
    def __init__(self, mass, pos, r, rotation, dampen=DEFAULT_DAMPEN, shadow=False):
        self.dampen = dampen
        self.r = r
        self.sphere = sphere(color = color.green, radius=r)
        self.pos = pos
        self.sphere.pos = pos
        self.mass = mass
        self.sphere.mass = mass
        self.v = vector(0, 0, 0)
        self.a = Mass.G
        theta, axis, origin = rotation
        self.sphere.rotate(angle=theta, axis=axis, origin=origin)
        self.pos = self.sphere.pos
        if shadow:
            self.shadow = cylinder(color=SHADOW_COLOR, radius=r, 
                                   axis=vector(0, 1, 0), length=SHADOW_HEIGHT,
                                   pos=vector(self.pos.x, 0, self.pos.z))
        else:
            self.shadow = None
        
    def update_a(self, Fs):
        a = vector(0,0,0)
        for F in Fs:
            a += F / self.mass
        self.a = a
    
    def update_mass(self, mass):
        self.mass = mass    
        self.sphere.mass = mass    
        
    def update_pos(self, pos):
        self.pos = pos
        self.sphere.pos = pos
        if self.shadow:
            self.shadow.pos.x = pos.x
            self.shadow.pos.z = pos.z
        
    def time_step(self, dt):
        self.v =  self.dampen * (self.v + self.a * dt)
        self.update_pos(self.pos + self.v * dt)

class MassLink:
    
    DEFAULT_K = 1000000
    DEFAULT_R = 0.005

    
    def __init__(self, mass1, mass2, k=DEFAULT_K, r=DEFAULT_R, shadow=False):
        i_mass1, mass1 = mass1
        i_mass2, mass2 = mass2
        self.mass_indices = (i_mass1, i_mass2)
        self.L_rest = mag(mass2.pos - mass1.pos)
        self.k = k
        self.rod = cylinder(pos=mass1.pos, axis=mass2.pos - mass1.pos, 
                            length=mag(mass2.pos - mass1.pos), radius=r)
        
        self.cmap = mpl.cm.bwr
        if shadow:
            self.shadow = self.get_shadow()
        else:
            self.shadow = None
    
    def get_shadow(self):
        v_rod = self.rod.axis
        v_proj = vector(v_rod.x, 0, v_rod.z)
        len_proj = mag(v_proj)
        dimensions = vector(len_proj, 0.0001, self.rod.radius)
        position = vector(self.rod.pos.x + self.rod.axis.x/2, 0, 
                               self.rod.pos.z + self.rod.axis.z/2)
        return box(color=SHADOW_COLOR, size=dimensions, axis=v_proj,
                   pos=position)
    
    def update_shadow(self):
        v_rod = self.rod.axis
        v_proj = vector(v_rod.x, 0, v_rod.z)
        len_proj = mag(v_proj)
        dimensions = vector(len_proj, SHADOW_HEIGHT, self.rod.radius)
        self.shadow.size=dimensions
        self.shadow.axis=v_proj
        self.shadow.pos=vector(self.rod.pos.x + self.rod.axis.x/2 + self.rod.radius/2, 
                               0, 
                               self.rod.pos.z + self.rod.axis.z/2 + self.rod.radius/2)
    
    def update_link(self, mass1, mass2):
        self.rod.pos = mass1.pos
        self.rod.axis = mass2.pos - mass1.pos
        self.rod.length = mag(mass2.pos - mass1.pos)
        if self.shadow:
            self.update_shadow()
        
    def get_force(self, mass):
        dL = self.L_rest - self.rod.length 
        dL_norm = 10**8 * dL + 0.5
        self.rod.color = vector(*self.cmap(dL_norm)[:3])
        # Compression:
        if dL < 0:
            src_to_dest = 1 if self.rod.pos == mass.pos else -1
        # Tension:
        elif dL > 0:
            src_to_dest = -1 if self.rod.pos == mass.pos else 1
        else:
            src_to_dest = 1
        F_dir = norm(self.rod.axis) * src_to_dest
        F_mag = self.k * abs(dL)
        return F_dir * F_mag
    
    def get_potential(self, mass):
        dL = self.L_rest - self.rod.length 
        F = self.get_force(mass)
        F_mag = mag(F)
        return (1/2) * F_mag * dL


def simulate(name, save_gif=True, n_frames=500, steps_per_frame=10, 
             plot_energy=True, single_cube=True, drop_height=0.2, 
             breathe=False, shadow=True):

    floor_y = 0
    floor_w = 4
    floor_h = 0.5
    floor_k = 200000
    
    wallB = box (pos=vector(0, -floor_h/2 + floor_y, 0), 
                 size=vector(-floor_w/2, floor_h, -floor_w/2),  color=color.white)
    
    scene.camera.pos = vector(0.12, 0.324885, 1.17934)
    
    cubes = []
    if single_cube:
        cube_ps = [vector(0, drop_height, 0.7)]   
    else:
        cube_ps = [vector(0, drop_height, 0.4),
                    vector(0.4, drop_height, 0.4),
                    vector(-0.4, drop_height, 0.4),
                    vector(-0.4, drop_height, 0.7),
                    vector(0.4, drop_height, 0.7),
                    vector(0, drop_height, 0.7)]
    
    for cube_p in cube_ps:
        mass_r = 0.01
        cube_m = 0.8
        cube_l = 0.1
        cube_rotation = (random.randint(360), 
                         vector(*[random.uniform(0, 1) for i in range(3)]), 
                         cube_p)
        masses = [Mass(cube_m/6, cube_p, mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(cube_l, 0, 0), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(0, cube_l, 0), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(0, 0, cube_l), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(cube_l, cube_l, 0), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(0, cube_l, cube_l), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(cube_l, 0, cube_l), mass_r, cube_rotation),
                  Mass(cube_m/6, cube_p + vector(cube_l, cube_l, cube_l), mass_r, cube_rotation)]
        
        
        rods = []
        for i, mass1 in enumerate(masses):
            for j, mass2 in enumerate(masses[i+1:]):
                rod = MassLink((i, mass1), (j+i+1, mass2))
                rods += [rod]
        Ls_rest = [rod.L_rest for rod in rods]
        cubes += [(masses, rods, Ls_rest)]
    
    KE = 0
    PE = 0
    PEs = []
    KEs = []
    Es = []
    
    dt = 0.0001
    T = 0
    Ts = []

    frame = 0
    t_per_frame = dt * steps_per_frame
    n_steps = 0
    
    
    while frame < n_frames:
        PE = 0
        KE = 0
        for cube in cubes:
            masses, rods, Ls_rest = cube
            for i, mass in enumerate(masses):
                default_Fs = [Mass.G * mass.mass]
                Fs = []
                if (mass.pos.y-mass.r) < floor_y:
                    Fs += [vector(0, floor_k * (floor_y - (mass.pos.y-mass.r)), 0)]
                    if plot_energy:
                        PE +=  (1/2) * floor_k * (floor_y - (mass.pos.y-mass.r))**2
                for rod in rods:
                    if i in rod.mass_indices:
                        Fs += [rod.get_force(mass)]
                        if plot_energy:
                            PE += rod.get_potential(mass)
                Fs += default_Fs
                mass.update_a(Fs)
                mass.time_step(dt)
            for L_rest, rod in zip(Ls_rest, rods):
                rod.update_link(*[masses[i] for i in rod.mass_indices])
                if breathe:
                    rod.L_rest = math.sin((100) * frame) * 0.02 * L_rest + 0.98 * L_rest
            if plot_energy:
                for mass in masses:
                    mass_h = mass.pos.y
                    mass_PE = mass.mass * mass_h * -Mass.G.y
                    mass_KE = (1/2) * mass.mass * mag(mass.v)**2
                    PE += mass_PE
                    KE += mass_KE
        if plot_energy:
            Ts += [T]
            PEs += [PE]
            KEs += [KE]
            Es += [PE + KE]
            T += dt
        if n_steps >= steps_per_frame:
            if save_gif:
                ImageGrab.grab(bbox=(100, 300, 1200, 1070)).save("ani_img/frame_{}.png".format(frame), "PNG")
            frame += 1
            n_steps = 1
        else:
            n_steps += 1
                
    if save_gif:
        img, *imgs = [Image.open(os.path.join("ani_img", f)) for f in os.listdir("ani_img")]
        img.save(fp="figs/{}.gif".format(name), format='GIF', append_images=imgs,
                 save_all=True, duration=t_per_frame, loop=1)
    
    if plot_energy:
        fig, ax = plt.subplots()
        ax.plot(Ts, PEs, label="PE")
        ax.plot(Ts, KEs, label="KE")
        ax.plot(Ts, Es, label="Total E")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.title("System energy")
        plt.legend(loc="lower right")
        plt.savefig("figs/system_energy_{}.png".format(name), dpi=300)

simulate("shadow", save_gif=False, n_frames=200, steps_per_frame=20, 
             plot_energy=False, single_cube=True, drop_height=0.05, breathe=True)
