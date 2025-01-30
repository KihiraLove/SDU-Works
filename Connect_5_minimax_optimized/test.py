import unittest

from Connect_5_minimax_optimized.board import Board
from Connect_5_minimax_optimized.bot import Bot


class TestBot(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.bot = Bot(self.board)

    def test_init(self):
        self.assertEqual(self.bot.board, self.board)
        self.assertEqual(self.bot.x_index_chains, [])
        self.assertEqual(self.bot.o_index_chains, [])

    def test_recalculate_chains(self):
        shifted_indexes = [{0, 19, 399, 418}, {0, 22, 44, 66}]
        self.bot.x_index_chains = [{0, 19, 380, 399}, {0, 21, 42, 63}]
        self.bot.o_index_chains = [{0, 19, 380, 399}, {0, 21, 42, 63}]
        self.bot.recalculate_chains()
        self.assertCountEqual(shifted_indexes, self.bot.x_index_chains)
        self.assertCountEqual(shifted_indexes, self.bot.o_index_chains)

    def test_add_new_chain_empty_list(self):
        self.bot.add_new_chain({0, 1}, True)
        self.bot.add_new_chain({2, 3}, False)
        self.assertEqual([{0, 1}], self.bot.x_index_chains)
        self.assertEqual([{2, 3}], self.bot.o_index_chains)

    def test_add_new_chain_existing_list(self):
        self.bot.x_index_chains = [{0, 1}]
        self.bot.o_index_chains = [{4, 5}]
        self.bot.add_new_chain({2, 3}, True)
        self.bot.add_new_chain({6, 7}, False)
        self.assertCountEqual([{0, 1}, {2, 3}], self.bot.x_index_chains)
        self.assertCountEqual([{4, 5}, {6, 7}], self.bot.o_index_chains)

    def test_add_last_move_with_overlap(self):
        move = (2, 20)
        solution = {19, 39, 59, 79}
        self.board.x_indexes.update({19, 59, 79})
        self.bot.x_index_chains.append({59, 79})
        self.bot.x_index_chains.append({19})
        self.bot.add_last_move(move, True)
        self.assertCountEqual(solution, self.bot.x_index_chains[0])

    def test_add_last_move_new_chain_empty_chain_list(self):
        moves = (2, 20)
        self.bot.add_last_move(moves, True)
        self.assertEqual({39}, self.bot.x_index_chains[0])

    def test_add_last_move_new_chain_no_matches(self):
        moves = (2, 20)
        self.bot.x_index_chains.append({37})
        self.board.add_index(37, True)
        self.bot.add_last_move(moves, True)
        self.assertCountEqual([{37}, {39}], self.bot.x_index_chains)

    def test_add_last_move_with_vet_chains(self):
        moves = (2, 20)
        self.bot.o_index_chains.append({38, 58, 78})
        self.board.add_index(38, False)
        self.board.add_index(58, False)
        self.board.add_index(78, False)
        self.bot.add_last_move(moves, True)
        self.assertCountEqual([{39}], self.bot.x_index_chains)
        self.assertCountEqual([{38, 58, 78}], self.bot.o_index_chains)

    def test_check_for_overlap_horizontal(self):
        self.bot.o_index_chains.append({84, 85, 86})
        self.bot.o_index_chains.append({82, 83, 84})
        self.bot.check_for_overlap([(0, 1), (1, 1)], False)
        self.assertCountEqual([{82, 83, 84, 85, 86}], self.bot.o_index_chains)

    def test_check_for_overlap_diagonal_down_up(self):
        self.bot.o_index_chains.append({84, 103, 122})
        self.bot.o_index_chains.append({122, 141, 160})
        self.bot.check_for_overlap([(0, 19), (1, 19)], False)
        self.assertCountEqual([{84, 103, 122, 141, 160}], self.bot.o_index_chains)

    def test_check_for_overlap_multiple_directions(self):
        self.bot.o_index_chains.append({46, 65, 84})
        self.bot.o_index_chains.append({84, 103, 122})
        self.bot.o_index_chains.append({84, 85, 86})
        self.bot.o_index_chains.append({82, 83, 84})
        self.bot.o_index_chains.append({44, 64, 84})
        self.bot.o_index_chains.append({84, 104, 124})
        self.bot.o_index_chains.append({42, 63, 84})
        self.bot.o_index_chains.append({84, 105, 126})
        changed_chains = [(0, 19), (1, 19), (2, 1), (3, 1), (4, 20), (5, 20), (6, 21), (7, 21)]
        solution = [{46, 65, 84, 103, 122}, {82, 83, 84, 85, 86}, {44, 64, 84, 104, 124}, {42, 63, 84, 105, 126}]
        self.bot.check_for_overlap(changed_chains, False)
        self.assertCountEqual(solution, self.bot.o_index_chains)

    def test_check_for_overlap_vertical(self):
        self.bot.o_index_chains.append({84, 104, 124})
        self.bot.o_index_chains.append({124, 144, 164})
        self.bot.check_for_overlap([(0, 20), (1, 20)], False)
        self.assertCountEqual([{84, 104, 124, 144, 164}], self.bot.o_index_chains)

    def test_check_for_overlap_diagonal_up_down(self):
        self.bot.o_index_chains.append({84, 105, 126})
        self.bot.o_index_chains.append({126, 147, 168})
        self.bot.check_for_overlap([(0, 21), (1, 21)], False)
        self.assertCountEqual([{84, 105, 126, 147, 168}], self.bot.o_index_chains)

    def test_check_for_overlap_new_chain_is_blocked(self):
        index = 150
        self.bot.x_index_chains = [{66, 87}, {192, 213}]
        self.bot.o_index_chains = [{108, 129, 150}, {150, 171}]
        self.board.x_indexes.update({66, 87, 192, 213})
        self.board.o_indexes.update({108, 129, 150, 171})
        self.bot.check_for_overlap([(0, 21), (1, 21)], False)
        self.assertEqual([], self.bot.o_index_chains)

    def test_is_index_in_row1(self):
        indexes_in_row_1 = [0, 1, 18, 19]
        indexes_on_edge = [20, 21, 38, 39]
        indexes_not_in_row1 = [40, 41, 58, 59]
        for index in indexes_in_row_1:
            self.assertTrue(self.bot.is_index_in_row1(index))
        for index in indexes_on_edge:
            self.assertFalse(self.bot.is_index_in_row1(index))
        for index in indexes_not_in_row1:
            self.assertFalse(self.bot.is_index_in_row1(index))

    def test_is_index_in_col1(self):
        indexes_in_row_1 = [0, 20, 360, 380]
        indexes_on_edge = [1, 21,  361, 381]
        indexes_not_in_row1 = [2, 22,  362, 382]
        for index in indexes_in_row_1:
            self.assertTrue(self.bot.is_index_in_col1(index))
        for index in indexes_on_edge:
            self.assertFalse(self.bot.is_index_in_col1(index))
        for index in indexes_not_in_row1:
            self.assertFalse(self.bot.is_index_in_col1(index))

    def test_is_chain_blocked_by_edge_blocked_vertical(self):
        direction = 20
        chain_neg_index = 0
        chain_pos_index = 20
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_not_blocked_vertical(self):
        direction = 20
        chain_neg_index = 360
        chain_pos_index = 380
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_blocked_horizontal(self):
        direction = 1
        chain_neg_index = 0
        chain_pos_index = 1
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_not_blocked_horizontal(self):
        direction = 1
        chain_neg_index = 18
        chain_pos_index = 19
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_blocked_diagonal_down_up(self):
        direction = 19
        # blocked by upper edge
        chain_neg_index = 19
        chain_pos_index = 38
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # blocked by left edge
        chain_neg_index = 21
        chain_pos_index = 40
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_not_blocked_diagonal_down_up(self):
        direction = 19
        # not blocked by right edge
        chain_neg_index = 39
        chain_pos_index = 58
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # not blocked by lower edge
        chain_neg_index = 381
        chain_pos_index = 362
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_blocked_diagonal_up_down(self):
        direction = 21
        # not blocked by left edge
        chain_neg_index = 20
        chain_pos_index = 41
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # not blocked by upper edge
        chain_neg_index = 1
        chain_pos_index = 22
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # not blocked by both edges
        chain_neg_index = 0
        chain_pos_index = 21
        self.assertTrue(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_is_chain_blocked_by_edge_not_diagonal_up_down(self):
        direction = 21
        # not blocked by both edges
        chain_neg_index = 378
        chain_pos_index = 399
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # not blocked by right edge
        chain_neg_index = 358
        chain_pos_index = 379
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))
        # not blocked by lower edge
        chain_neg_index = 377
        chain_pos_index = 398
        self.assertFalse(self.bot.is_chain_blocked_by_edge(direction, chain_neg_index, chain_pos_index))

    def test_check_for_open_chains(self):
        self.bot.x_index_chains = [{0}, {2, 3}, {5, 6, 7}]
        self.bot.o_index_chains = [{20}, {22, 23}, {25, 26, 27}]
        self.assertIsNone(self.bot.check_for_open_chains(4, True))
        self.assertIsNone(self.bot.check_for_open_chains(4, False))
        self.bot.add_new_chain({9, 10, 11, 12}, True)
        self.bot.add_new_chain({29, 30, 31, 32}, False)
        self.assertEqual(3, self.bot.check_for_open_chains(4, True))
        self.assertEqual(3, self.bot.check_for_open_chains(4, False))

    def test_add_last_move(self):
        index = 45
        self.bot.x_index_chains = [{24}, {64}, {66, 87}, {26, 46, 66}]
        self.board.x_indexes.update({24, 26, 45, 46, 64, 66, 87})
        self.bot.add_last_move((3, 6), True)
        expected = [{24, 45, 66, 87}, {45, 46}, {26, 45, 64}, {26, 46, 66}]
        actual = self.bot.x_index_chains
        self.assertCountEqual(expected, actual)

    def test_check_for_4_move(self):
        self.bot.x_index_chains = [{20, 40, 60, 80}]
        self.board.x_indexes.update({20, 40, 60, 80})
        self.assertIn(self.bot.check_for_4_move(True), [(1, 1), (6, 1)])

    def test_check_for_4_move_blocked_by_edge(self):
        self.bot.x_index_chains = [{0, 20, 40, 60}]
        self.board.x_indexes.update({0, 20, 40, 60})
        self.assertEqual((5, 1), self.bot.check_for_4_move(True))

    def test_check_for_4_move_blocked_by_opponent(self):
        self.bot.x_index_chains = [{20, 40, 60, 80}]
        self.board.x_indexes.update({20, 40, 60, 80})
        self.bot.o_index_chains = [{0}]
        self.board.o_indexes.update({0})
        self.assertEqual((6, 1), self.bot.check_for_4_move(True))

    def test_check_for_4_move_blocked_from_both_sides(self):
        # This can never happen, but we test it in case a bug makes it happen
        # Function doesn't raise errors anymore, this test will fail every time
        self.bot.x_index_chains = [{20, 40, 60, 80}]
        self.board.x_indexes.update({20, 40, 60, 80})
        self.bot.o_index_chains = [{0}, {100}]
        self.board.o_indexes.update({0, 100})
        with self.assertRaises(RuntimeError):
            move = self.bot.check_for_4_move(True)
            print(move)

    def test_vet_closed_chains(self):
        self.bot.x_index_chains = [{87, 108, 129}]
        self.bot.o_index_chains = [{66}, {150}]
        self.board.x_indexes.update({87, 108, 129})
        self.board.o_indexes.update({66, 150})
        self.bot.vet_closed_chains(150, 129, True)
        self.assertEqual([], self.bot.x_index_chains)

    def test_vet_closed_chains_1_long_not_closed(self):
        self.bot.x_index_chains = [{0}, {19}, {380}, {399}]
        self.board.x_indexes = {0, 19, 380, 399}
        self.board.o_indexes = {1, 18, 381, 398}
        self.bot.vet_closed_chains(1, 0, True)
        self.bot.vet_closed_chains(18, 19, True)
        self.bot.vet_closed_chains(381, 380, True)
        self.bot.vet_closed_chains(398, 399, True)
        self.assertEqual([{0}, {19}, {380}, {399}], self.bot.x_index_chains)

    def test_vet_closed_chains_1_long_closed(self):
        self.bot.x_index_chains = [{0}, {19}, {380}, {399}]
        self.board.x_indexes = {0, 19, 380, 399}
        self.board.o_indexes = {1, 18, 20, 21, 38, 39, 360, 361, 378, 379, 381, 398}
        self.bot.vet_closed_chains(1, 0, True)
        self.bot.vet_closed_chains(18, 19, True)
        self.bot.vet_closed_chains(381, 380, True)
        self.bot.vet_closed_chains(398, 399, True)
        self.assertEqual([], self.bot.x_index_chains)

    def test_get_available_moves_around_1_long_chains(self):
        self.bot.x_index_chains = [{0}, {19}, {380}, {399}]
        self.bot.o_index_chains = [{1}, {18}, {381}, {398}]
        self.board.x_indexes = {0, 19, 380, 399}
        self.board.o_indexes = {1, 18, 381, 398}
        expected = sorted([2, 17, 20, 21, 22, 37, 38, 39, 360, 361, 362, 377, 378, 379, 382, 397])
        actual = sorted(self.bot.get_available_moves_around_1_long_chains())
        self.assertEqual(expected, actual)


class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_init(self):
        self.assertEqual(self.board.o_indexes, set())
        self.assertEqual(self.board.x_indexes, set())
        self.assertEqual(self.board.size, 20)

    def test_enlarge(self):
        self.board.o_indexes.add(0)
        self.board.o_indexes.add(19)
        self.board.x_indexes.add(380)
        self.board.x_indexes.add(399)
        self.board.enlarge()
        self.assertEqual(self.board.o_indexes, {0, 19})
        self.assertEqual(self.board.x_indexes, {399, 418})
        self.assertEqual(self.board.size, 21)

    def test_shift_indexes(self):
        indexes = {0, 19, 380, 399}
        shifted_indexes = {0, 19, 399, 418}
        self.assertEqual(self.board.shift_indexes(indexes), shifted_indexes)

    def test_calculate_index_from_position(self):
        x = [1, 1, 20, 20]
        y = [1, 20, 1, 20]
        indexes = [0, 19, 380, 399]
        for i in range(len(x)):
            self.assertEqual(self.board.calculate_index_from_position(x[i], y[i]), indexes[i])

    def test_calculate_position_from_index(self):
        x = [1, 1, 20, 20]
        y = [1, 20, 1, 20]
        indexes = [0, 19, 380, 399]
        for i in range(len(x)):
            self.assertEqual(self.board.calculate_position_from_index(indexes[i]), (x[i], y[i]))

    def test_add_index(self):
        player_x = True
        player_o = False
        indexes = {0, 19, 380, 399}
        for index in indexes:
            self.board.add_index(index, player_x)
            self.board.add_index(index, player_o)
        self.assertEqual(self.board.x_indexes, indexes)
        self.assertEqual(self.board.o_indexes, indexes)

    def test_set_position(self):
        x = [1, 1, 20, 20]
        y = [1, 20, 1, 20]
        x_over_size = 21
        y_over_size = 21
        player_x = True
        player_o = False
        indexes = {0, 19, 380, 399}
        shifted_indexes = {0, 19, 399, 418, 440}
        for i in range(len(x)):
            self.assertFalse(self.board.set_position(x[i], y[i], player_x))
            self.assertFalse(self.board.set_position(x[i], y[i], player_o))

        self.assertEqual(self.board.x_indexes, indexes)
        self.assertEqual(self.board.o_indexes, indexes)

        self.assertTrue(self.board.set_position(x_over_size, y_over_size, player_x))
        self.assertEqual(self.board.x_indexes, shifted_indexes)

    def test_is_position_valid_empty(self):
        x = [1, 1, 20, 20]
        y = [1, 20, 1, 20]
        for i in range(len(x)):
            self.assertTrue(self.board.is_position_valid_from_pos(x[i], y[i]))

    def test_is_position_valid_occupied(self):
        player_x = True
        x = [1, 1, 20, 20]
        y = [1, 20, 1, 20]
        for i in range(len(x)):
            self.board.set_position(x[i], y[i], player_x)

        for i in range(len(x)):
            self.assertFalse(self.board.is_position_valid_from_pos(x[i], y[i]))

    def test_is_position_valid_out_of_bounds(self):
        player_x = True
        x = [0, 0, 22, 22]
        y = [0, 22, 0, 22]
        for i in range(len(x)):
            self.board.set_position(x[i], y[i], player_x)

        for i in range(len(x)):
            self.assertFalse(self.board.is_position_valid_from_pos(x[i], y[i]))

    def test_is_index_occupied(self):
        x_indexes = {0, 19, 380, 399}
        o_indexes = {1, 18, 381, 398}
        for i in x_indexes.union(o_indexes):
            self.assertFalse(self.board.is_index_occupied(i))
        for i in x_indexes:
            self.board.x_indexes.add(i)
        for i in o_indexes:
            self.board.o_indexes.add(i)
        for i in x_indexes.union(o_indexes):
            self.assertTrue(self.board.is_index_occupied(i))

    def test_is_index_in_indexes_for_player(self):
        x_index = 0
        o_index = 1
        player_x = True
        player_o = False

        self.assertFalse(self.board.is_index_in_indexes_for_player(x_index, player_x))
        self.assertFalse(self.board.is_index_in_indexes_for_player(o_index, player_o))

        self.board.x_indexes.add(x_index)
        self.board.o_indexes.add(o_index)
        self.assertTrue(self.board.is_index_in_indexes_for_player(x_index, player_x))
        self.assertTrue(self.board.is_index_in_indexes_for_player(o_index, player_o))

    def test_get_neighbours(self):
        neighbours = {1, 20, 21, 379, 398, 378, 18, 38, 39, 360, 361, 381}
        indexes = {0, 19, 380, 399}
        player_x = True
        player_o = False
        for neighbour in neighbours:
            self.board.x_indexes.add(neighbour)
            self.board.o_indexes.add(neighbour)
        self.assertEqual({1, 20, 21}, self.board.get_neighbours(0, player_x))
        self.assertEqual({18, 38, 39}, self.board.get_neighbours(19, player_x))
        self.assertEqual({360, 361, 381}, self.board.get_neighbours(380, player_x))
        self.assertEqual({379, 398, 378}, self.board.get_neighbours(399, player_x))
        self.assertEqual({1, 20, 21}, self.board.get_neighbours(0, player_o))
        self.assertEqual({18, 38, 39}, self.board.get_neighbours(19, player_o))
        self.assertEqual({360, 361, 381}, self.board.get_neighbours(380, player_o))
        self.assertEqual({379, 398, 378}, self.board.get_neighbours(399, player_o))

    def test_move(self):
        x = [1, 21]
        y = [1, 21]
        player_x = True
        self.assertFalse(self.board.move(x[0], y[0], player_x)[1])
        with self.assertRaises(IndexError):
            self.board.move(x[0], y[0], player_x)
        self.assertTrue(self.board.move(x[1], y[1], player_x)[1])

    def test_check_for_win_horizontal(self):
        player_x = True
        chain = {1, 2, 3, 4}
        x = [1, 1, 1]
        y = [1, 3, 5]
        index = 0
        for i in chain:
            self.board.x_indexes.add(i)

        self.assertFalse(self.board.check_for_win(1, 2, player_x))
        self.board.x_indexes.add(index)
        for i in range(len(x)):
            self.assertTrue(self.board.check_for_win(x[i], y[i], player_x))

    def test_check_for_win_vertical(self):
        player_x = True
        chain = {20, 40, 60, 80}
        x = [1, 3, 5]
        y = [1, 1, 1]
        index = 0
        for i in chain:
            self.board.x_indexes.add(i)

        self.assertFalse(self.board.check_for_win(2, 1, player_x))
        self.board.x_indexes.add(index)
        for i in range(len(x)):
            self.assertTrue(self.board.check_for_win(x[i], y[i], player_x))

    def test_check_for_win_diagonal_up_down(self):
        player_x = True
        chain = {21, 42, 63, 84}
        x = [1, 3, 5]
        y = [1, 3, 5]
        index = 0
        for i in chain:
            self.board.x_indexes.add(i)

        self.assertFalse(self.board.check_for_win(2, 2, player_x))
        self.board.x_indexes.add(index)
        for i in range(len(x)):
            self.assertTrue(self.board.check_for_win(x[i], y[i], player_x))

    def test_check_for_win_diagonal_down_up(self):
        player_x = True
        chain = {61, 42, 23, 4}
        x = [5, 3, 1]
        y = [1, 3, 5]
        index = 80
        for i in chain:
            self.board.x_indexes.add(i)

        self.assertFalse(self.board.check_for_win(4, 2, player_x))
        self.board.x_indexes.add(index)
        for i in range(len(x)):
            self.assertTrue(self.board.check_for_win(x[i], y[i], player_x))

    def test_print_board(self):
        self.board.print_board()
