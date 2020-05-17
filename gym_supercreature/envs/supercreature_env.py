import math
from sympy.geometry import *

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

FRICTION = 2.5

BODY_L = 2
BODY_W = 0.3
LIMB_L = [0.85, 0.85, 0.65, 0.65]
LIMB_W = [0.1, 0.1, 0.15, 0.15]
ARM_Y_SHIFT = 0.1
LEG_ANGLE = math.radians(15)
ARM_ANGLE = math.radians(35)

JOINT_SPEED = 4
MOTOR_TORQUE = 80

TOTAL_LENGTH = 150

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

        self.body_fixture = fixtureDef(
            shape=polygonShape(vertices=[]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0020,
            maskBits=0x001,
            restitution=0.0
        )
        self.terrain_fixture = fixtureDef(
            shape=polygonShape(vertices=[]),
            friction=FRICTION,
            categoryBits=0x0001
        )

        self.action_space = spaces.Box(np.array([-1] * 8), np.array([1] * 8), dtype=np.float32)
        high = np.array([np.inf] * 36)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        terrain_h = 0.2 * H
        self.terrain_fixture.shape.vertices = [(-100000, 0), (-100000, terrain_h), (100000, terrain_h), (100000, 0)]
        self.terrain = self.world.CreateStaticBody(fixtures=self.terrain_fixture)
        self.terrain.color1 = (0.0, 1.0, 0.0)

        base_limb_points = [
            [Point2D(-LIMB_W[0] / 2, 0), Point2D(LIMB_W[0] / 2, 0),
             Point2D(LIMB_W[0] / 2, -LIMB_L[0]), Point2D(-LIMB_W[0] / 2, -LIMB_L[0])],
            [Point2D(-LIMB_W[1] / 2, -LIMB_L[0]), Point2D(LIMB_W[1] / 2, -LIMB_L[0]),
             Point2D(LIMB_W[1] / 2, -LIMB_L[0] - LIMB_L[1]), Point2D(-LIMB_W[1] / 2, -LIMB_L[0] - LIMB_L[1])],
            [Point2D(-LIMB_W[2] / 2, 0), Point2D(LIMB_W[2] / 2, 0),
             Point2D(LIMB_W[2] / 2, -LIMB_L[2]), Point2D(-LIMB_W[2] / 2, -LIMB_L[2])],
            [Point2D(-LIMB_W[3] / 2, -LIMB_L[2]), Point2D(LIMB_W[3] / 2, -LIMB_L[2]),
             Point2D(LIMB_W[3] / 2, -LIMB_L[2] - LIMB_L[3]), Point2D(-LIMB_W[3] / 2, -LIMB_L[2] - LIMB_L[3])],
        ]
        left_leg_points = [[p.rotate(LEG_ANGLE) for p in base_limb_points[i]] for i in range(2)]
        right_leg_points = [[p.rotate(LEG_ANGLE) for p in base_limb_points[i]] for i in range(2)]
        left_arm_points = [[p.rotate(ARM_ANGLE) for p in base_limb_points[i]] for i in range(2, 4)]
        right_arm_points = [[p.rotate(ARM_ANGLE) for p in base_limb_points[i]] for i in range(2, 4)]

        legHeight = -min([float(p.y) for p in left_leg_points[1]])
        bodyY = float(terrain_h + legHeight + 0.1)
        bodyX = W * 0.1

        self.body_fixture.shape.vertices = [
            (bodyX, bodyY),
            (bodyX + BODY_W, bodyY),
            (bodyX + BODY_W, bodyY + BODY_L),
            (bodyX, bodyY + BODY_L)
        ]
        self.body = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.body.color1 = (1.0, 0.0, 0.0)
        self.body.contacts_ground = False

        leg_offset = Point2D(bodyX + BODY_W / 2, bodyY)
        arm_offset = Point2D(bodyX + BODY_W / 2, bodyY + BODY_L * (1 - ARM_Y_SHIFT))
        all_points = left_leg_points + right_leg_points + left_arm_points + right_arm_points
        self.limbs = []
        for i in range(len(all_points)):
            offset = leg_offset if i < 4 else arm_offset
            all_points[i] = [p + offset for p in all_points[i]]
            self.body_fixture.shape.vertices = [(float(p.x), float(p.y)) for p in all_points[i]]
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
                anchor=(float(offset.x), float(offset.y))
            ))

        self.contactListener = ContactDetector(self)
        self.world.contactListener = self.contactListener

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
            reward -= 0.00035 * MOTOR_TORQUE * np.clip(np.abs(action[i]), 0, 1)

        self.steps += 1

        self.world.Step(1.0/FPS, 6*30, 2*30)

        new_body_pos = (self.body.worldCenter.x, self.body.worldCenter.x)
        reward += (new_body_pos[0] - old_body_pos[0]) * 130 / SCALE
        reward += (new_body_pos[1] - old_body_pos[1]) * 500 / SCALE
        done = new_body_pos[0] > TOTAL_LENGTH

        if self.body.contacts_ground:
            reward = -200
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

        self.scroll = self.body.position.x // W * W

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
        print(rs)
        env.render()
        if done:
            rs = 0
            env.reset()
    env.close()
