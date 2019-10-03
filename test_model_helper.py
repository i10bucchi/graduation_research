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
        return_values1 = return_values1.values

        # ポイント初期化確認
        expected = np.zeros(NUM_PLAYERS)
        actual = return_values1[:, COL_P]
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
        arg = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg[:, COL_Qa00] = 1
        return_values = model_helper.get_members_action(arg, self.parameter)
        
        expected = np.zeros(NUM_MEMBERS)
        actual = return_values[:, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)
        
        expected = np.zeros(NUM_MEMBERS)
        actual = return_values[:, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)

        expected = np.zeros(NUM_MEMBERS)
        actual = return_values[:, COL_ANUM]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)

        # argmax_a(Q) = [0,1]
        arg = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg[:, COL_Qa01] = 1
        return_values = model_helper.get_members_action(arg, self.parameter)
        
        expected = np.zeros(NUM_MEMBERS)
        actual = return_values[:, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)

        expected = np.ones(NUM_MEMBERS)
        actual = return_values[:, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)

        expected = np.ones(NUM_MEMBERS)
        actual = return_values[:, COL_ANUM]
        self.assertEquals(np.sum(expected == actual), NUM_MEMBERS)

    def test_get_leader_action(self):
        # 乱数による決定が入るためepsilon=0の場合のテストを行う

        # argmax_a(Q) = [0,0]
        arg = np.zeros(NUM_COLUMN)
        arg[COL_Qa00] = 1
        return_values = model_helper.get_leader_action(arg, self.parameter)

        expected = 0
        actual = return_values[COL_APC]
        self.assertEqual(expected, actual)

        expected = 0
        actual = return_values[COL_APS]
        self.assertEqual(expected, actual)

        expected = 0
        actual = return_values[COL_ANUM]
        self.assertEqual(expected, actual)

        # argmax_a(Q) = [0,1]
        arg = np.zeros(NUM_COLUMN)
        arg[COL_Qa01] = 1
        return_values = model_helper.get_leader_action(arg, self.parameter)

        expected = 0
        actual = return_values[COL_APC]
        self.assertEqual(expected, actual)

        expected = 1
        actual = return_values[COL_APS]
        self.assertEqual(expected, actual)

        expected = 1
        actual = return_values[COL_ANUM]
        self.assertEqual(expected, actual)

    def test_get_members_gain(self):
        # 利得最小値を1にするための切片値
        min_r = self.parameter['punish_size'] * 2 + 1
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p1arg1 = arg1
        p1arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])
        p1arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])
        
        p1arg2 = arg2
        p1arg2[COL_APC] = 0
        p1arg2[COL_APS] = 0

        expected = np.array([6, 6, 6, 6, 6]) + min_r
        actual = model_helper.get_members_gain(p1arg1, p1arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p2arg1 = arg1
        p2arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p2arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])
        
        p2arg2 = arg2
        p2arg2[COL_APC] = 0
        p2arg2[COL_APS] = 0

        expected = np.array([6.4, 6.4, 12.4, 12.4, 12.4]) + min_r
        actual = model_helper.get_members_gain(p2arg1, p2arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p3arg1 = arg1
        p3arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p3arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])
        
        p3arg2 = arg2
        p3arg2[COL_APC] = 0
        p3arg2[COL_APS] = 0

        expected = np.array([16, 16, 16, 16, 16]) + min_r
        actual = model_helper.get_members_gain(p3arg1, p3arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p4arg1 = arg1
        p4arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])
        p4arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])
        
        p4arg2 = arg2
        p4arg2[COL_APC] = 1
        p4arg2[COL_APS] = 0

        expected = np.array([4 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p4arg1, p4arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p5arg1 = arg1
        p5arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p5arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])
        
        p5arg2 = arg2
        p5arg2[COL_APC] = 1
        p5arg2[COL_APS] = 0

        expected = np.array([6.4, 6.4, 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p5arg1, p5arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p6arg1 = arg1
        p6arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p6arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])
        
        p6arg2 = arg2
        p6arg2[COL_APC] = 1
        p6arg2[COL_APS] = 0

        expected = np.array([18, 18, 18, 18, 18]) + min_r
        actual = model_helper.get_members_gain(p6arg1, p6arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p7arg1 = arg1
        p7arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])
        p7arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])
        
        p7arg2 = arg2
        p7arg2[COL_APC] = 0
        p7arg2[COL_APS] = 1

        expected = np.array([4, 4, 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p7arg1, p7arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p8arg1 = arg1
        p8arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p8arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])
        
        p8arg2 = arg2
        p8arg2[COL_APC] = 0
        p8arg2[COL_APS] = 1

        expected = np.array([8.4 - self.parameter['punish_size'], 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p8arg1, p8arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p9arg1 = arg1
        p9arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p9arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])
        
        p9arg2 = arg2
        p9arg2[COL_APC] = 0
        p9arg2[COL_APS] = 1

        expected = np.array([16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p9arg1, p9arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p10arg1 = arg1
        p10arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])
        p10arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])
        
        p10arg2 = arg2
        p10arg2[COL_APC] = 1
        p10arg2[COL_APS] = 1

        expected = np.array([4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p10arg1, p10arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p11arg1 = arg1
        p11arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p11arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])
        
        p11arg2 = arg2
        p11arg2[COL_APC] = 1
        p11arg2[COL_APS] = 1

        expected = np.array([6.4, 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p11arg1, p11arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)


        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p12arg1 = arg1
        p12arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p12arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])
        
        p12arg2 = arg2
        p12arg2[COL_APC] = 1
        p12arg2[COL_APS] = 1

        expected = np.array([16, 16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']]) + min_r
        actual = model_helper.get_members_gain(p12arg1, p12arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members)

    def test_get_leaders_gain(self):
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_members = 5
        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p1arg1 = arg1
        p1arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])

        p1arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])

        p1arg2 = arg2
        p1arg2[COL_APC] = 0
        p1arg2[COL_APS] = 0

        # 利得最小値を1にするための切片値
        min_r = self.parameter['cost_punish'] * num_members * 2 + 1
        expected = min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, self.parameter)
        self.assertEqual(expected, actual)


        p1arg1 = arg1
        p1arg1[:, COL_AC] = np.array([1, 0, 0, 0, 0])
        p1arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])

        p1arg2 = arg2
        p1arg2[COL_APC] = 0
        p1arg2[COL_APS] = 0

        expected = 2 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, self.parameter)
        self.assertEqual(expected, actual)

        p1arg1 = arg1
        p1arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p1arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])

        p1arg2 = arg2
        p1arg2[COL_APC] = 0
        p1arg2[COL_APS] = 0

        expected = 4 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, self.parameter)
        self.assertEqual(expected, actual)

        p1arg1 = arg1
        p1arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p1arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])

        p1arg2 = arg2
        p1arg2[COL_APC] = 0
        p1arg2[COL_APS] = 0

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, self.parameter)
        self.assertEqual(expected, actual)        


        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p2arg1 = arg1
        p2arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])

        p2arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])

        p2arg2 = arg2
        p2arg2[COL_APC] = 1
        p2arg2[COL_APS] = 0

        expected = -10 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, self.parameter)
        self.assertEqual(expected, actual)


        p2arg1 = arg1
        p2arg1[:, COL_AC] = np.array([1, 0, 0, 0, 0])
        p2arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])

        p2arg2 = arg2
        p2arg2[COL_APC] = 1
        p2arg2[COL_APS] = 0

        expected = -6 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, self.parameter)
        self.assertEqual(expected, actual)

        p2arg1 = arg1
        p2arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p2arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])

        p2arg2 = arg2
        p2arg2[COL_APC] = 1
        p2arg2[COL_APS] = 0

        expected = -2 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, self.parameter)
        self.assertEqual(expected, actual)

        p2arg1 = arg1
        p2arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p2arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])

        p2arg2 = arg2
        p2arg2[COL_APC] = 1
        p2arg2[COL_APS] = 0

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, self.parameter)
        self.assertEqual(expected, actual) 


        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p3arg1 = arg1
        p3arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])

        p3arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])

        p3arg2 = arg2
        p3arg2[COL_APC] = 0
        p3arg2[COL_APS] = 1

        expected = -10 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, self.parameter)
        self.assertEqual(expected, actual)


        p3arg1 = arg1
        p3arg1[:, COL_AC] = np.array([1, 0, 0, 0, 0])
        p3arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])

        p3arg2 = arg2
        p3arg2[COL_APC] = 0
        p3arg2[COL_APS] = 1

        expected = -6 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, self.parameter)
        self.assertEqual(expected, actual)

        p3arg1 = arg1
        p3arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p3arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])

        p3arg2 = arg2
        p3arg2[COL_APC] = 0
        p3arg2[COL_APS] = 1

        expected = -2 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, self.parameter)
        self.assertEqual(expected, actual)

        p3arg1 = arg1
        p3arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p3arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])

        p3arg2 = arg2
        p3arg2[COL_APC] = 0
        p3arg2[COL_APS] = 1

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, self.parameter)
        self.assertEqual(expected, actual) 


        arg1 = np.zeros((num_members, NUM_COLUMN))
        arg2 = np.zeros(NUM_COLUMN)

        p4arg1 = arg1
        p4arg1[:, COL_AC] = np.array([0, 0, 0, 0, 0])

        p4arg1[:, COL_AS] = np.array([0, 0, 0, 0, 0])

        p4arg2 = arg2
        p4arg2[COL_APC] = 1
        p4arg2[COL_APS] = 1

        expected = -20 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, self.parameter)
        self.assertEqual(expected, actual)


        p4arg1 = arg1
        p4arg1[:, COL_AC] = np.array([1, 0, 0, 0, 0])
        p4arg1[:, COL_AS] = np.array([1, 0, 0, 0, 0])

        p4arg2 = arg2
        p4arg2[COL_APC] = 1
        p4arg2[COL_APS] = 1

        expected = -14 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, self.parameter)
        self.assertEqual(expected, actual)

        p4arg1 = arg1
        p4arg1[:, COL_AC] = np.array([1, 1, 0, 0, 0])
        p4arg1[:, COL_AS] = np.array([1, 1, 0, 0, 0])

        p4arg2 = arg2
        p4arg2[COL_APC] = 1
        p4arg2[COL_APS] = 1

        expected = -8 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, self.parameter)
        self.assertEqual(expected, actual)

        p4arg1 = arg1
        p4arg1[:, COL_AC] = np.array([1, 1, 1, 1, 1])
        p4arg1[:, COL_AS] = np.array([1, 1, 1, 1, 1])

        p4arg2 = arg2
        p4arg2[COL_APC] = 1
        p4arg2[COL_APS] = 1

        expected = 10 + min_r
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, self.parameter)
        self.assertEqual(expected, actual) 
    
    def test_learning_members(self):
        arg1 = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg1[:, COL_P] = np.array(range(NUM_MEMBERS))
        arg1[:, COL_ANUM] = np.full(NUM_MEMBERS, 1)

        return_value = model_helper.learning_members(arg1, self.parameter)

        actual = return_value
        expected = arg1
        expected[:, COL_Qa01] = 0.1 * np.array(range(NUM_MEMBERS))

        self.assertEquals(np.sum(actual == expected), NUM_MEMBERS*NUM_COLUMN)

    def test_learning_leader(self):
        arg1 = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg1[:, COL_P_LOG] = np.array(range(NUM_MEMBERS))
        arg1[:, COL_ANUM] = np.full(NUM_MEMBERS, 1)
        arg2 = np.zeros(NUM_COLUMN)
        arg2[COL_P] = 10
        arg2[COL_ANUM] = 1 
        arg4 = [0, 0]

        return_value = model_helper.learning_leader(arg1, arg2, self.parameter, arg4)

        actual = return_value
        expected = arg2

        expected[COL_Qa01] = 0.1 * arg2[COL_P]
        self.assertEquals(np.sum(actual == expected), NUM_COLUMN)

        arg1 = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg1[:, COL_P_LOG] = np.array(range(NUM_MEMBERS))
        arg1[:, COL_ANUM] = np.full(NUM_MEMBERS, 1)
        arg2 = np.zeros(NUM_COLUMN)
        arg2[COL_P] = 10
        arg2[COL_ANUM] = 1 
        arg4 = [0, 1]

        return_value = model_helper.learning_leader(arg1, arg2, self.parameter, arg4)

        actual = return_value
        expected = arg2

        expected[COL_Qa01] = 0.1 *  arg2[COL_P] / LEADER_SAMPLING_TERM
        self.assertEquals(np.sum(actual == expected), NUM_COLUMN)

        arg1 = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg1[:, COL_P_LOG] = np.array(range(NUM_MEMBERS))
        arg1[:, COL_ANUM] = np.full(NUM_MEMBERS, 1)
        arg2 = np.zeros(NUM_COLUMN)
        arg2[COL_P] = 10
        arg2[COL_ANUM] = 1 
        arg4 = [1, 0]

        return_value = model_helper.learning_leader(arg1, arg2, self.parameter, arg4)

        actual = return_value
        expected = arg2

        expected[COL_Qa01] = 0.1 * ( np.mean(np.array(range(NUM_MEMBERS))) * arg2[COL_P] )
        self.assertEquals(np.sum(actual == expected), NUM_COLUMN)

        arg1 = np.zeros((NUM_MEMBERS, NUM_COLUMN))
        arg1[:, COL_P_LOG] = np.array(range(NUM_MEMBERS))
        arg1[:, COL_ANUM] = np.full(NUM_MEMBERS, 1)
        arg2 = np.zeros(NUM_COLUMN)
        arg2[COL_P] = 10
        arg2[COL_ANUM] = 1 
        arg4 = [1, 1]

        return_value = model_helper.learning_leader(arg1, arg2, self.parameter, arg4)

        actual = return_value
        expected = arg2

        expected[COL_Qa01] = 0.1 * ( np.mean(np.array(range(NUM_MEMBERS))) * arg2[COL_P] ) / LEADER_SAMPLING_TERM
        self.assertEquals(np.sum(actual == expected), NUM_COLUMN)

    def test_get_players_rule(self):
        # 乱数による決定が入るためepsilon=0の場合のテストを行う

        # argmax_a(Q) = [0,0]
        arg = np.zeros((NUM_PLAYERS, NUM_COLUMN))
        arg[:, COL_Qr00] = 1
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
        arg1[:, COL_RREWARD] = np.ones(NUM_PLAYERS)
        arg2 = 2
        
        actual = model_helper.learning_rule(arg1, arg2, alpha=0.1)
        expected = np.zeros((NUM_PLAYERS, 4))
        expected[:, 2] = 0.1
        self.assertEquals(np.sum(actual == expected), NUM_PLAYERS*4)
    
if __name__ == "__main__":
    unittest.main()