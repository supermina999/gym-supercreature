import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering


FPS = 50
SCALE = 30

VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

JOINT_SPEED = 4
MOTOR_TORQUE = 80


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def rotate(self, alpha):
        return Point(self.x * math.cos(alpha) - self.y * math.sin(alpha),
                     self.x * math.sin(alpha) + self.y * math.cos(alpha))


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        for limb in self.env.limbs:
            if limb in [contact.fixtureA.body, contact.fixtureB.body]:
                limb.contacts_ground = True
        if self.env.body in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.body.contacts_ground = True

    def EndContact(self, contact):
        for limb in self.env.limbs:
            if limb in [contact.fixtureA.body, contact.fixtureB.body]:
                limb.contacts_ground = False
        if self.env.body in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.body.contacts_ground = False



class SupercreatureEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.fps': FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.scroll = 0

        self.world = Box2D.b2World()
        self.terrain = None
        self.body = None
        self.joints = []
        self.last_action = [0] * 8

        self.action_space = spaces.Box(np.array([-1] * 8), np.array([1] * 8), dtype=np.float32)
        high = np.array([np.inf] * 48)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        #self.body.ApplyForceToCenter((self.np_random.uniform(-10000, 10000), self.np_random.uniform(-10000, 10000)), False)

        self.steps = 0

    def cleanup(self):
        if self.terrain is None:
            return

        self.world.DestroyBody(self.terrain)
        self.terrain = None
        self.world.DestroyBody(self.body)
        self.body = None
        for limb in self.limbs:
            self.world.DestroyBody(limb)
        self.limbs = []
        self.joints = []
        self.world.contactListener = self.contactListener = None

    def reset(self):
        self.cleanup()

        self.density = 7
        self.friction = self.np_random.uniform(0.5, 1)
        self.height_reward = 500
        self.body_l = self.np_random.uniform(1.8, 2.5)
        self.body_w = self.np_random.uniform(0.25, 0.35)
        self.arm_y_shift = self.np_random.uniform(0.05, 0.15)
        self.limb_l = self.np_random.uniform(0.6, 0.9, 4)
        self.limb_w = self.np_random.uniform(0.075, 0.175, 4)

        self.body_fixture = fixtureDef(
            shape=polygonShape(vertices=[]),
            density=self.density,
            friction=self.friction,
            categoryBits=0x0020,
            maskBits=0x001,
            restitution=0.0
        )
        self.terrain_fixture = fixtureDef(
            shape=polygonShape(vertices=[]),
            friction=1.0,
            categoryBits=0x0001
        )

        self.terrain_h = 0.2 * H
        self.terrain_fixture.shape.vertices = [(-100000, 0), (-100000, self.terrain_h), (100000, self.terrain_h), (100000, 0)]
        self.terrain = self.world.CreateStaticBody(fixtures=self.terrain_fixture)
        self.terrain.color1 = (0.0, 1.0, 0.0)

        self.contactListener = ContactDetector(self)
        self.world.contactListener = self.contactListener

        base_limb_points = [
            [Point(-self.limb_w[0] / 2, 0), Point(self.limb_w[0] / 2, 0),
             Point(self.limb_w[0] / 2, -self.limb_l[0]), Point(-self.limb_w[0] / 2, -self.limb_l[0])],
            [Point(-self.limb_w[1] / 2, -self.limb_l[0]), Point(self.limb_w[1] / 2, -self.limb_l[0]),
             Point(self.limb_w[1] / 2, -self.limb_l[0] - self.limb_l[1]), Point(-self.limb_w[1] / 2, -self.limb_l[0] - self.limb_l[1])],
            [Point(-self.limb_w[2] / 2, 0), Point(self.limb_w[2] / 2, 0),
             Point(self.limb_w[2] / 2, -self.limb_l[2]), Point(-self.limb_w[2] / 2, -self.limb_l[2])],
            [Point(-self.limb_w[3] / 2, -self.limb_l[2]), Point(self.limb_w[3] / 2, -self.limb_l[2]),
             Point(self.limb_w[3] / 2, -self.limb_l[2] - self.limb_l[3]), Point(-self.limb_w[3] / 2, -self.limb_l[2] - self.limb_l[3])],
        ]
        angles = self.np_random.uniform(-math.pi / 4, math.pi / 4, 4)
        left_leg_points = [[p.rotate(angles[0]) for p in base_limb_points[i]] for i in range(2)]
        right_leg_points = [[p.rotate(angles[1]) for p in base_limb_points[i]] for i in range(2)]
        left_arm_points = [[p.rotate(angles[2]) for p in base_limb_points[i]] for i in range(2, 4)]
        right_arm_points = [[p.rotate(angles[3]) for p in base_limb_points[i]] for i in range(2, 4)]

        legHeight = -min([p.y for p in left_leg_points[1]])
        bodyY = float(self.terrain_h + legHeight + 0.1)
        bodyX = W * 0.1

        self.body_fixture.shape.vertices = [
            (bodyX, bodyY),
            (bodyX + self.body_w, bodyY),
            (bodyX + self.body_w, bodyY + self.body_l),
            (bodyX, bodyY + self.body_l)
        ]
        self.body = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.body.color1 = (1.0, 0.0, 0.0)
        self.body.contacts_ground = False

        leg_offset = Point(bodyX + self.body_w / 2, bodyY)
        arm_offset = Point(bodyX + self.body_w / 2, bodyY + self.body_l * (1 - self.arm_y_shift))
        all_points = left_leg_points + right_leg_points + left_arm_points + right_arm_points
        self.limbs = []
        for i in range(len(all_points)):
            offset = leg_offset if i < 4 else arm_offset
            all_points[i] = [p + offset for p in all_points[i]]
            self.body_fixture.shape.vertices = [(p.x, p.y) for p in all_points[i]]
            limb = self.world.CreateDynamicBody(fixtures=self.body_fixture)
            limb.color1 = (1.0, 0.0, 0.0)
            limb.contacts_ground = False
            self.limbs.append(limb)

        for i in range(len(self.limbs)):
            offset = (all_points[i][0] + all_points[i][1]) / 2
            bodyA = self.body if i % 2 == 0 else self.limbs[i - 1]
            self.joints.append(self.world.CreateRevoluteJoint(
                bodyA=bodyA,
                bodyB=self.limbs[i],
                anchor=(offset.x, offset.y)
            ))

        self.body.linearVelocity.x = 0
        self.body.linearVelocity.y = 0
        self.body.angularVelocity = 0
        self.body.transform = ((0, 0), 0)
        self.last_action = [0] * 8

        for limb in self.limbs:
            limb.linearVelocity.x = 0
            limb.linearVelocity.y = 0
            limb.angularVelocity = 0
            limb.transform = ((0, 0), 0)

        self.body.contacts_ground = False

        for limb in self.limbs:
            limb.contacts_ground = False

        self.steps = 0

        self.world.ClearForces()
        self.body.ApplyForceToCenter((self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5)), True)

        return self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        state = [
            self.friction,
            self.body_l,
            self.body_w,
            self.arm_y_shift,
        ]
        state += self.limb_l.tolist()
        state += self.limb_w.tolist()

        state += [
            self.body.linearVelocity.x,
            self.body.linearVelocity.y,
            self.body.angle,
            self.body.angularVelocity,
        ]
        for i in range(len(self.joints)):
            state.append(self.joints[i].angle)
            state.append(self.joints[i].speed)
            state.append(1.0 if self.limbs[i].contacts_ground else 0.0)
        for i in range(len(self.last_action)):
            state.append(self.last_action[i])

        return np.array(state)

    def step(self, action):
        old_body_pos = (self.body.worldCenter.x, self.body.worldCenter.x)
        self.last_action = action

        reward = 0

        for i in range(len(self.joints)):
            self.joints[i].motorEnabled = True
            self.joints[i].motorSpeed = float(JOINT_SPEED * np.sign(action[i]))
            self.joints[i].maxMotorTorque = float(MOTOR_TORQUE * np.clip(np.abs(action[i]), 0, 1))
            #reward -= 0.00035 * MOTOR_TORQUE * np.clip(np.abs(action[i]), 0, 1)

        self.steps += 1

        self.world.Step(1.0/FPS, 6*30, 2*30)

        new_body_pos = (self.body.worldCenter.x, self.body.worldCenter.x)
        reward += (new_body_pos[0] - old_body_pos[0]) * 130 / SCALE
        reward += (new_body_pos[1] - old_body_pos[1]) * self.height_reward / SCALE
        done = False

        body_angle = self.body.angle
        reward += 3 - math.fabs(body_angle)
        if body_angle < -math.radians(60) or body_angle > math.radians(60):
            reward = -2000
            done = True

        if self.body.contacts_ground:
            reward = -2000
            done = True

        if self.steps > 1000:
            done = True

        return self.get_state(), reward, done, {}

    def drawBody(self, body):
        transform = body.fixtures[0].body.transform
        path = [transform * v for v in body.fixtures[0].shape.vertices]
        self.viewer.draw_polygon(path, color=body.color1)
        #path.append(path[0])
        #self.viewer.draw_polyline(path, color=body.color2)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

        self.scroll = self.body.worldCenter.x // W * W

        self.viewer.set_bounds(self.scroll, W + self.scroll, 0, H)
        self.viewer.draw_polygon([
            (self.scroll, 0),
            (self.scroll + W, 0),
            (self.scroll + W, H),
            (self.scroll, H)
        ], color=(0.9, 0.9, 1.0))

        self.drawBody(self.terrain)
        self.drawBody(self.body)
        for limb in self.limbs:
            self.drawBody(limb)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

import gym.wrappers

if __name__=="__main__":
    env = SupercreatureEnv()
    #env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force = True)
    env.reset()

    rs = 0
    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step([1, 0, 0, 0, 0, 0, 0, 0])
        rs += reward
        #print(rs)
        env.render()
        if done:
            rs = 0
            env.reset()
    env.close()
