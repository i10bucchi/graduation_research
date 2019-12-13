#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import model_helper
import numpy as np
from config import *

class TestModel(unittest.TestCase):
    def setUp(self):
        self.parameter = {
            'cost_cooperate':   4,
            'cost_support':     2,
            'cost_punish':      2,
            'power_social':     4,
            'punish_size':      8,
            'alpha':            0.1,
            'epsilon':          0,
        }

    def test_generate_players(self):
        return_values1 = model_helper.generate_players()

        # ポイント初期化確認
        expected = np.zeros(NUM_PLAYERS)
        actual = return_values1[:, COL_GAME_REWARD]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        # ポイントログ初期化確認
        expected = np.zeros(NUM_PLAYERS)
        actual = return_values1[:, COL_P_LOG]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        # 行動の初期化確認
        expected = np.full(NUM_PLAYERS, -1)
        actual = return_values1[:, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        expected = np.full(NUM_PLAYERS, -1)
        actual = return_values1[:, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        expected = np.full(NUM_PLAYERS, -1)
        actual = return_values1[:, COL_APC]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        expected = np.full(NUM_PLAYERS, -1)
        actual = return_values1[:, COL_APS]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        expected = np.full(NUM_PLAYERS, -1)
        actual = return_values1[:, COL_ANUM]
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        # 乱数区間確認
        expected_area = [0, 1]
        actual = return_values1[:, COL_Qa00]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_PLAYERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_PLAYERS)

        expected_area = [0, 1]
        actual = return_values1[:, COL_Qa01]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_PLAYERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_PLAYERS)

        expected_area = [0, 1]
        actual = return_values1[:, COL_Qa10]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_PLAYERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_PLAYERS)

        expected_area = [0, 1]
        actual = return_values1[:, COL_Qa11]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_PLAYERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_PLAYERS)

    def test_get_members_action(self):
        # 乱数による決定が入るためepsilon=0の場合のテストを行う

        # argmax_a(Q) = [0,0]
        arg = np.zeros((NUM_MEMBERS, 4))
        arg[:, 0] = 1
        actual1, actual2 = model_helper.get_members_action(arg, self.parameter)
        
        expected1 = np.zeros((NUM_MEMBERS, 2))
        expected2 = np.zeros(NUM_MEMBERS)

        self.assertEquals(np.sum(expected1 == actual1), NUM_MEMBERS*2)
        self.assertEquals(np.sum(expected2 == actual2), NUM_MEMBERS)

        # argmax_a(Q) = [0,1]
        arg = np.zeros((NUM_MEMBERS, 4))
        arg[:, 1] = 1
        actual1, actual2 = model_helper.get_members_action(arg, self.parameter)
        
        expected1 = np.zeros((NUM_MEMBERS, 2))
        expected1[:, 1] = 1
        expected2 = np.ones(NUM_MEMBERS)

        self.assertEquals(np.sum(expected1 == actual1), NUM_MEMBERS*2)
        self.assertEquals(np.sum(expected2 == actual2), NUM_MEMBERS)

    def test_get_leader_action(self):
        # 乱数による決定が入るためepsilon=0の場合のテストを行う

        # argmax_a(Q) = [0,0]
        arg = np.zeros(4)
        arg[0] = 1
        actual1, actual2 = model_helper.get_leader_action(arg, self.parameter)

        expected1 = np.array([0, 0])
        expected2 = 0

        self.assertEqual(np.sum(expected1 == actual1), 2)
        self.assertEqual(expected2, actual2)

        # argmax_a(Q) = [0,1]
        arg = np.zeros(4)
        arg[1] = 1
        actual1, actual2 = model_helper.get_leader_action(arg, self.parameter)

        expected1 = np.array([0, 1])
        expected2 = 1

        self.assertEqual(np.sum(expected1 == actual1), 2)
        self.assertEqual(expected2, actual2)

    def test_get_members_gain(self):
        # 利得最小値を1にするための切片値
        min_r = self.parameter['punish_size'] * 2 + 1
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_members = 5

        p1arg1 = np.array([0, 0, 0, 0, 0])
        p1arg2 = np.array([0, 0, 0, 0, 0])
        
        p1arg3 = 0
        p1arg4 = 0

        expected = np.array([6, 6, 6, 6, 6]) + min_r
        actual = model_helper.get_members_gain(p1arg1, p1arg2, p1arg3, p1arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p2arg1 = np.array([1, 1, 0, 0, 0])
        p2arg2 = np.array([1, 1, 0, 0, 0])
        
        p2arg3 = 0
        p2arg4 = 0

        expected = np.array([6.4, 6.4, 12.4, 12.4, 12.4]) + min_r
        actual = model_helper.get_members_gain(p2arg1, p2arg2, p2arg3, p2arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p3arg1 = np.array([1, 1, 1, 1, 1])
        p3arg2 = np.array([1, 1, 1, 1, 1])
        
        p3arg3 = 0
        p3arg4 = 0

        expected = np.array([16, 16, 16, 16, 16]) + min_r
        actual = model_helper.get_members_gain(p3arg1, p3arg2, p3arg3, p3arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p4arg1 = np.array([0, 0, 0, 0, 0])
        p4arg2 = np.array([1, 0, 0, 0, 0])
        
        p4arg3 = 1
        p4arg4 = 0

        expected = np.array([4 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p4arg1, p4arg2, p4arg3, p4arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p5arg1 = np.array([1, 1, 0, 0, 0])
        p5arg2 = np.array([1, 1, 1, 1, 1])
        
        p5arg3 = 1
        p5arg4 = 0

        expected = np.array([6.4, 6.4, 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p5arg1, p5arg2, p5arg3, p5arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p6arg1 = np.array([1, 1, 1, 1, 1])
        p6arg2 = np.array([0, 0, 0, 0, 0])
        
        p6arg3 = 1
        p6arg4 = 0

        expected = np.array([18, 18, 18, 18, 18]) + min_r
        actual = model_helper.get_members_gain(p6arg1, p6arg2, p6arg3, p6arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p7arg1 = np.array([0, 0, 0, 0, 0])
        p7arg2 = np.array([1, 1, 0, 0, 0])
        
        p7arg3 = 0
        p7arg4 = 1

        expected = np.array([4, 4, 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p7arg1, p7arg2, p7arg3, p7arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p8arg1 = np.array([1, 1, 0, 0, 0])
        p8arg2 = np.array([0, 0, 0, 0, 0])
        
        p8arg3 = 0
        p8arg4 = 1

        expected = np.array([8.4 - self.parameter['punish_size'], 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p8arg1, p8arg2, p8arg3, p8arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p9arg1 = np.array([1, 1, 1, 1, 1])
        p9arg2 = np.array([1, 0, 0, 0, 0])
        
        p9arg3 = 0
        p9arg4 = 1

        expected = np.array([16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p9arg1, p9arg2, p9arg3, p9arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p10arg1 = np.array([0, 0, 0, 0, 0])
        p10arg2 = np.array([1, 1, 1, 1, 1])
        
        p10arg3 = 1
        p10arg4 = 1

        expected = np.array([4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p10arg1, p10arg2, p10arg3, p10arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p11arg1 = np.array([1, 1, 0, 0, 0])
        p11arg2 = np.array([1, 0, 0, 0, 0])
        
        p11arg3 = 1
        p11arg4 = 1

        expected = np.array([6.4, 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p11arg1, p11arg2, p11arg3, p11arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5

        p12arg1 = np.array([1, 1, 1, 1, 1])
        p12arg2 = np.array([1, 1, 0, 0, 0])
        
        p12arg3 = 1
        p12arg4 = 1

        expected = np.array([16, 16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p12arg1, p12arg2, p12arg3, p12arg4,  self.parameter, num_members=num_members)
        self.assertEquals(np.sum(expected == actual), num_members)

    def test_get_leaders_gain(self):
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_members = 5

        
        p1arg1 = np.array([0, 0, 0, 0, 0])
        p1arg2 = np.array([0, 0, 0, 0, 0])
        p1arg3 = 0
        p1arg4 = 0

        # 利得最小値を1にするための切片値
        min_r = self.parameter['cost_punish'] * num_members * 2 + 1
        expected = min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, p1arg3, p1arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)


        
        p1arg1 = np.array([1, 0, 0, 0, 0])
        p1arg2 = np.array([1, 0, 0, 0, 0])
        p1arg3 = 0
        p1arg4 = 0

        expected = 2 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, p1arg3, p1arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p1arg1 = np.array([1, 1, 0, 0, 0])
        p1arg2 = np.array([1, 1, 0, 0, 0])
        p1arg3 = 0
        p1arg4 = 0

        expected = 4 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, p1arg3, p1arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p1arg1 = np.array([1, 1, 1, 1, 1])
        p1arg2 = np.array([1, 1, 1, 1, 1])
        p1arg3 = 0
        p1arg4 = 0

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, p1arg3, p1arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)        

        
        p2arg1 = np.array([0, 0, 0, 0, 0])
        p2arg2 = np.array([0, 0, 0, 0, 0])
        p2arg3 = 1
        p2arg4 = 0

        expected = -10 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, p2arg3, p2arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)


        p2arg1 = np.array([1, 0, 0, 0, 0])
        p2arg2 = np.array([1, 0, 0, 0, 0])
        p2arg3 = 1
        p2arg4 = 0

        expected = -6 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, p2arg3, p2arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p2arg1 = np.array([1, 1, 0, 0, 0])
        p2arg2 = np.array([1, 1, 0, 0, 0])
        p2arg3 = 1
        p2arg4 = 0

        expected = -2 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, p2arg3, p2arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p2arg1 = np.array([1, 1, 1, 1, 1])
        p2arg2 = np.array([1, 1, 1, 1, 1])
        p2arg3 = 1
        p2arg4 = 0

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, p2arg3, p2arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual) 

        
        p3arg1 = np.array([0, 0, 0, 0, 0])
        p3arg2 = np.array([0, 0, 0, 0, 0])
        p3arg3 = 0
        p3arg4 = 1

        expected = -10 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, p3arg3, p3arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)


        
        p3arg1 = np.array([1, 0, 0, 0, 0])
        p3arg2 = np.array([1, 0, 0, 0, 0])
        p3arg3 = 0
        p3arg4 = 1

        expected = -6 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, p3arg3, p3arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p3arg1 = np.array([1, 1, 0, 0, 0])
        p3arg2 = np.array([1, 1, 0, 0, 0])
        p3arg3 = 0
        p3arg4 = 1

        expected = -2 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, p3arg3, p3arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p3arg1 = np.array([1, 1, 1, 1, 1])
        p3arg2 = np.array([1, 1, 1, 1, 1])
        p3arg3 = 0
        p3arg4 = 1

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, p3arg3, p3arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual) 



        p4arg1 = np.array([0, 0, 0, 0, 0])
        p4arg2 = np.array([0, 0, 0, 0, 0])
        p4arg3 = 1
        p4arg4 = 1

        expected = -20 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, p4arg3, p4arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)


        
        p4arg1 = np.array([1, 0, 0, 0, 0])
        p4arg2 = np.array([1, 0, 0, 0, 0])
        p4arg3 = 1
        p4arg4 = 1

        expected = -14 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, p4arg3, p4arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p4arg1 = np.array([1, 1, 0, 0, 0])
        p4arg2 = np.array([1, 1, 0, 0, 0])
        p4arg3 = 1
        p4arg4 = 1

        expected = -8 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, p4arg3, p4arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual)

        
        p4arg1 = np.array([1, 1, 1, 1, 1])
        p4arg2 = np.array([1, 1, 1, 1, 1])
        p4arg3 = 1
        p4arg4 = 1

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, p4arg3, p4arg4, self.parameter, num_members=num_members)
        self.assertEqual(expected, actual) 
    
    def test_learning_members(self):
        arg1 = np.zeros((NUM_MEMBERS, 4))
        arg2 = np.array(range(NUM_MEMBERS))
        arg3 = np.full(NUM_MEMBERS, 1)

        actual = model_helper.learning_members(arg1, arg2, arg3, self.parameter)
        expected = np.zeros((NUM_MEMBERS, 4))
        expected[:, 1] = 0.1 * np.array(range(NUM_MEMBERS))

        self.assertEquals(np.sum(actual == expected), NUM_MEMBERS*4)

    def test_learning_leader(self):
        
        arg1 = np.array(range(NUM_MEMBERS))
        arg2 = np.zeros(4)
        arg3 = 1
        arg4 = 10
        arg6 = [0, 0]

        actual = model_helper.learning_leader(arg1, arg2, arg3, arg4, self.parameter, arg6)
        expected = np.zeros(4)
        expected[arg3] = 0.1 * arg4
        self.assertEquals(np.sum(actual == expected), 4)

        
        arg1 = np.array(range(NUM_MEMBERS))
        arg2 = np.zeros(4)
        arg3 = 1 
        arg4 = 10
        arg6 = [0, 1]

        actual = model_helper.learning_leader(arg1, arg2, arg3, arg4, self.parameter, arg6)
        expected = np.zeros(4)
        expected[arg3] = 0.1 *  arg4 / LEADER_SAMPLING_TERM
        # ここで少数の四捨五入をするのは微小な計算違いを起こすため
        self.assertEquals(np.sum(np.round(actual, decimals=6) == expected), 4)

        
        arg1 = np.array(range(NUM_MEMBERS))
        arg2 = np.zeros(4)
        arg3 = 1 
        arg4 = 10
        arg6 = [1, 0]

        actual = model_helper.learning_leader(arg1, arg2, arg3, arg4, self.parameter, arg6)
        expected = np.zeros(4)
        expected[arg3] = 0.1 * ( np.mean(np.array(range(NUM_MEMBERS))) * arg4 )
        self.assertEquals(np.sum(actual == expected), 4)

        
        arg1 = np.array(range(NUM_MEMBERS))
        arg2 = np.zeros(4)
        arg3 = 1 
        arg4 = 10
        arg6 = [1, 1]

        actual = model_helper.learning_leader(arg1, arg2, arg3, arg4, self.parameter, arg6)
        expected = np.zeros(4)
        expected[arg3] = 0.1 * ( np.mean(np.array(range(NUM_MEMBERS))) * arg4 ) / LEADER_SAMPLING_TERM
        self.assertEquals(np.sum(actual == expected), 4)

    def test_get_players_rule(self):
        # 乱数による決定が入るためepsilon=0の場合のテストを行う

        # argmax_a(Q) = [0,0]
        arg = np.zeros((NUM_PLAYERS, NUM_COLUMN))
        arg[:, COL_QrLEADER] = 1
        actual = model_helper.get_players_rule(arg, epshilon=0)

        expected = np.zeros(NUM_PLAYERS)
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)

        # argmax_a(Q) = [0,1]
        arg = np.zeros((NUM_PLAYERS, NUM_COLUMN))
        arg[:, COL_Qr01] = 1
        actual = model_helper.get_players_rule(arg, epshilon=0)

        expected = np.ones(NUM_PLAYERS)
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)
    
    def test_get_gaming_rule(self):
        num_players = 7
        # 最大票数が選択されるかのテスト
        arg = np.zeros((num_players, NUM_COLUMN))
        arg[:, COL_RNUM] = np.array([0, 1, 1, 2, 2, 2, 2])

        actual = model_helper.get_gaming_rule(arg)
        expected = 2
        self.assertEqual(expected, actual)

        # 票が割れた場合-1を返すかのテスト
        arg = np.zeros((num_players, NUM_COLUMN))
        arg[:, COL_RNUM] = np.array([0, 0, 1, 1, 2, 2, 3])

        actual = model_helper.get_gaming_rule(arg)
        expected = -1
        self.assertEqual(expected, actual)
    
    def test_get_rule_gain(self):
        arg = np.zeros((NUM_PLAYERS, NUM_COLUMN))
        arg[:, COL_Qa00] = np.ones(NUM_PLAYERS)

        actual = model_helper.get_rule_gain(arg)
        expected = np.ones(NUM_PLAYERS)
        self.assertEquals(np.sum(expected == actual), NUM_PLAYERS)
    
    def test_learning_rule(self):
        arg1 = np.zeros((NUM_PLAYERS, NUM_COLUMN))
        arg1[:, COL_ROLE_REWARD] = np.ones(NUM_PLAYERS)
        arg2 = 2
        
        actual = model_helper.learning_rule(arg1, arg2, alpha=0.1)
        expected = np.zeros((NUM_PLAYERS, 4))
        expected[:, 2] = 0.1
        self.assertEquals(np.sum(actual == expected), NUM_PLAYERS*4)
    
if __name__ == "__main__":
    unittest.main()