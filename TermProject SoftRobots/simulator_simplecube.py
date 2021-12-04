from vpython import scene, vector, color, box, rate


scene.caption = """To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""


floor_y = 0
floor_w = 10
floor_h = 0.2
floor_k = 200000

wallB = box (pos=vector(0, -floor_h + floor_y, 0), 
             size=vector(-floor_w/2, floor_h, -floor_w/2),  color = color.blue)


cube_m = 5
cube_p = vector(-0.5, 0.75, 0.5)
cube_l = 0.5

cube = box (color = color.green, size=vector(cube_l, cube_l, cube_l), 
            make_trail=True, retain=200)

cube.pos = cube_p
cube.mass = cube_m


cube_v = vector(0, 0, 0)
cube_a = vector(0, -9.8, 0)
dt = 0.0005
while True:
    rate(200)
    cube_v = cube_v + cube_a * dt
    cube.pos = cube.pos + (cube_v) * dt
    if floor_y >= cube.pos.y:
        F_k = -floor_k * cube.pos.y
        a_k = F_k / cube.mass
        cube_v.y += a_k * dt
