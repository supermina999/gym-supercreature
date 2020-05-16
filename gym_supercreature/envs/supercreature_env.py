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
LEG_L = 1.7
LEG_W = 0.1
ARM_L = 1.3
ARM_W = 0.15
ARM_Y_SHIFT = 0.1
HEAD_R = BODY_W / 3
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

        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        high = np.array([np.inf] * 16)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        terrain_h = 0.2 * H
        self.terrain_fixture.shape.vertices = [(-10000, 0), (-10000, terrain_h), (10000, terrain_h), (10000, 0)]
        self.terrain = self.world.CreateStaticBody(fixtures=self.terrain_fixture)
        self.terrain.color1 = (0.0, 1.0, 0.0)

        base_leg_points = [Point2D(-LEG_W / 2, 0), Point2D(LEG_W / 2, 0),
                           Point2D(LEG_W / 2, -LEG_L), Point2D(-LEG_W / 2, -LEG_L)]
        left_leg_points = [p.rotate(-LEG_ANGLE) for p in base_leg_points]
        right_leg_points = [p.rotate(LEG_ANGLE) for p in base_leg_points]

        base_arm_points = [Point2D(-ARM_W / 2, 0), Point2D(ARM_W / 2, 0),
                           Point2D(ARM_W / 2, -ARM_L), Point2D(-ARM_W / 2, -ARM_L)]
        left_arm_points = [p.rotate(-ARM_ANGLE) for p in base_arm_points]
        right_arm_points = [p.rotate(ARM_ANGLE) for p in base_arm_points]

        legHeight = -min([float(p.y) for p in left_leg_points])
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
        self.body_fixture.shape.vertices = [(float(p.x + leg_offset.x), float(p.y + leg_offset.y)) for p in
                                            left_leg_points]
        self.left_leg = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.left_leg.color1 = (1.0, 0.0, 0.0)
        self.left_leg.contacts_ground = True

        self.body_fixture.shape.vertices = [(float(p.x + leg_offset.x), float(p.y + leg_offset.y)) for p in
                                            right_leg_points]
        self.right_leg = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.right_leg.color1 = (1.0, 0.0, 0.0)
        self.right_leg.contacts_ground = True

        arm_offset = Point2D(bodyX + BODY_W / 2, bodyY + BODY_L * (1 - ARM_Y_SHIFT))
        self.body_fixture.shape.vertices = [(float(p.x + arm_offset.x), float(p.y + arm_offset.y)) for p in
                                            left_arm_points]
        self.left_arm = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.left_arm.color1 = (1.0, 0.0, 0.0)
        self.left_arm.contacts_ground = False

        self.body_fixture.shape.vertices = [(float(p.x + arm_offset.x), float(p.y + arm_offset.y)) for p in
                                            right_arm_points]
        self.right_arm = self.world.CreateDynamicBody(fixtures=self.body_fixture)
        self.right_arm.color1 = (1.0, 0.0, 0.0)
        self.right_arm.contacts_ground = False

        # self.body.ApplyForceToCenter((self.np_random.uniform(-5, 5), 0), True)

        leg_offset = (float(leg_offset.x), float(leg_offset.y))
        arm_offset = (float(arm_offset.x), float(arm_offset.y))

        self.joints.append(self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.left_leg,
            anchor=leg_offset
        ))

        self.joints.append(self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.right_leg,
            anchor=leg_offset
        ))

        self.joints.append(self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.left_arm,
            anchor=arm_offset
        ))

        self.joints.append(self.world.CreateRevoluteJoint(
            bodyA=self.body,
            bodyB=self.right_arm,
            anchor=arm_offset
        ))

        self.limbs = [self.left_leg, self.right_leg, self.left_arm, self.right_arm]

        self.contactListener = ContactDetector(self)
        self.world.contactListener = self.contactListener

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
        self.left_leg = self.right_leg = self.left_arm = self.right_arm = None
        self.joints = []
        self.world.contactListener = self.contactListener = None

    def reset(self):
        self.body.linearVelocity.x = 0
        self.body.linearVelocity.y = 0
        self.body.angularVelocity = 0
        self.body.transform = ((0, 0), 0)

        for limb in self.limbs:
            limb.linearVelocity.x = 0
            limb.linearVelocity.y = 0
            limb.angularVelocity = 0
            limb.transform = ((0, 0), 0)

        self.body.contacts_ground = False
        self.left_leg.contacts_ground = True
        self.right_leg.contacts_ground = True
        self.left_arm.contacts_ground = False
        self.right_arm.contacts_ground = False

        self.steps = 0

        self.world.ClearForces()

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
        for i in range(4):
            state.append(self.joints[i].angle)
            state.append(self.joints[i].speed)
            state.append(1.0 if self.limbs[i].contacts_ground else 0.0)

        return np.array(state)

    def step(self, action):
        old_body_x = self.body.position.x

        reward = 0

        for i in range(4):
            self.joints[i].motorEnabled = True
            self.joints[i].motorSpeed = float(JOINT_SPEED * np.sign(action[i]))
            self.joints[i].maxMotorTorque = float(MOTOR_TORQUE * np.clip(np.abs(action[i]), 0, 1))
            reward -= 0.00035 * MOTOR_TORQUE * np.clip(np.abs(action[i]), 0, 1)

        self.steps += 1

        self.world.Step(1.0/FPS, 6*30, 2*30)

        reward += (self.body.position.x - old_body_x) * 130 / SCALE
        done = self.body.position.x > TOTAL_LENGTH

        if self.body.contacts_ground or (self.steps > 2000 and self.body.position.x < 40):
            reward = -200
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
        self.drawBody(self.left_leg)
        self.drawBody(self.right_leg)
        self.drawBody(self.left_arm)
        self.drawBody(self.right_arm)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

import gym.wrappers

if __name__=="__main__":
    env = SupercreatureEnv()
    env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force = True)
    env.reset()

    rs = 0
    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step([0, 1, 0, 0])
        rs += reward
        print(rs)
        env.render()
        if done:
            env.reset()
    env.close()
