import random
from board import Board
import copy


class Node:
    def __init__(self, is_player_x, index=None, bot=None):
        self.value = float("-inf")
        if index is not None:
            self.step = bot.board.calculate_position_from_index(index)
            if bot.board.is_index_occupied(index):
                bot.add_last_move(self.step, is_player_x)
                bot.board.move(self.step[0], self.step[1], is_player_x)
        else:
            self.step = None
        self.is_player_x = is_player_x
        self.bot = bot

    def set_value(self, val):
        self.value = val

    def get_board(self):
        return self.bot.board

    def set_step(self, step):
        self.step = step

    def get_player(self):
        return self.is_player_x


class Bot:
    def __init__(self, board):
        self.board = board
        self.x_index_chains = []
        self.o_index_chains = []
        self.step = None
        self.three_x_count = 0
        self.three_o_count = 0
        self.four_x_count = 0
        self.four_o_count = 0

    def drop_invalid_moves(self, indexes):
        valid_moves = []
        for index in indexes:
            if not self.board.is_index_occupied(index):
                valid_moves.append(index)
        return valid_moves

    def collect_possible_indexes(self, is_player_x):
        """
        :param is_player_x: if is_maximizing_player:
                                False
                            else:
                                True
        :return: all the considered possible moves in a given state
        """
        possible_indexes = []
        # consider open 4 'O' or 'X'
        possible_indexes.append(self.check_for_4_move(is_player_x))
        if possible_indexes[0] is not None:
            # this is horrible but the check_for_4_move return a move and not an index...
            return [self.board.calculate_index_from_position(possible_indexes[0][0], possible_indexes[0][1])]
        else:
            possible_indexes.pop(0)
        # consider open 4 'X' or 'O'
        possible_indexes.append(self.check_for_4_move(not is_player_x))
        if possible_indexes[0] is not None:
            return [self.board.calculate_index_from_position(possible_indexes[0][0], possible_indexes[0][1])]
        else:
            possible_indexes.pop(0)
        # consider open 3
        possible_indexes.extend(list(self.find_double_open_3_chains(is_player_x)))
        # consider open 3
        if len(possible_indexes) == 0:
            possible_indexes.extend(list(self.find_double_open_3_chains(not is_player_x)))
        # consider every other possible moves
        if len(possible_indexes) == 0:
            possible_indexes.extend(list(self.get_all_chain_edge_indexes(3, is_player_x)))
            possible_indexes.extend(list(self.get_all_chain_edge_indexes(3, not is_player_x)))
            possible_indexes.extend(list(self.get_all_chain_edge_indexes(2, is_player_x)))
            possible_indexes.extend(list(self.get_all_chain_edge_indexes(2, not is_player_x)))
            possible_indexes.extend(self.get_available_moves_around_1_long_chains())
        possible_indexes = list(filter(lambda item: item is not None, possible_indexes))
        possible_indexes = self.drop_duplicates(possible_indexes)
        possible_indexes = self.drop_invalid_moves(possible_indexes)
        return possible_indexes

    def heuristic(self):
        """
        Heuristic function for finding the heuristic value of a node.
        :param node: The node to find the heuristic value for.
        :return: The heuristic value of a node.
        """
        node_value = 0
        # collect points for all 4 chains
        node_value += len(self.get_all_open_chains(4, False)) * 16
        node_value += self.four_o_count * 16
        node_value -= len(self.get_all_open_chains(4, True)) * 16
        node_value -= self.four_x_count * 16
        # collect points for 3 emp-emp
        node_value += len(self.find_double_open_3_chains(False)) * 8
        node_value -= len(self.find_double_open_3_chains(True)) * 8
        # collect points for all 3 chains
        node_value += len(list(
            filter(lambda item: item is not None, self.get_all_chain_edge_indexes(3, False)))) * 4
        node_value += self.three_o_count * 4
        node_value -= len(list(
            filter(lambda item: item is not None, self.get_all_chain_edge_indexes(3, True)))) * 4
        node_value -= self.three_x_count * 4
        node_value += len(list(filter(lambda item: item is not None, self.get_all_open_chains(2, False)))) * 2
        return node_value

    @staticmethod
    def drop_duplicates(list_in):
        return list(set(list_in))

    def minimax(self, node, depth, is_maximizing_player, alpha, beta):
        # if someone won return a corresponding inf value else return heuristic
        if node.step is not None:
            if node.get_board().check_for_win(node.step[0], node.step[1], node.get_player()):
                if node.get_player() is False:
                    return float('+inf')
                else:
                    return float('-inf')
            elif depth == 7:
                return node.bot.heuristic()

        if is_maximizing_player:
            best_val = float('-inf')
            all_possible_indexes = copy.deepcopy(node.bot.collect_possible_indexes(False))
            for one_index in all_possible_indexes:
                new_node = Node(False, one_index, bot=copy.deepcopy(node.bot))
                value = self.minimax(new_node, depth + 1, False, alpha, beta)
                best_val = max(best_val, value)
                if value >= best_val and depth == 0:
                    self.step = one_index
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val

        else:
            best_val = float('+inf')
            all_possible_indexes = copy.deepcopy(node.bot.collect_possible_indexes(True))
            for one_index in all_possible_indexes:
                new_node = Node(True, one_index, bot=copy.deepcopy(node.bot))
                value = self.minimax(new_node, depth + 1, True, alpha, beta)
                best_val = min(best_val, value)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val

    # TODO: Cache subtree
    # TODO: add a function to check for off the board moves when choosing a move, to have the correct index

    def recalculate_chains(self):
        """
        Recalculates the saved chains after enlargement
        """
        self.recalculate_chain(True)
        self.recalculate_chain(False)

    def recalculate_chain(self, is_player_x):
        """
        Recalculate a specific chain after enlargement
        :param is_player_x: boolean indicating whether the player is X or not
        """
        for index_chain in self.x_index_chains if is_player_x else self.o_index_chains:
            new_index_chain = self.board.shift_indexes(index_chain)
            index_chain.clear()
            index_chain.update(new_index_chain)

    def add_last_move(self, move, is_player_x):
        """
        Add the last move to the cache
        :param move: (x, y) tuple containing the coordinates of the last move
        :param is_player_x: boolean indicating whether the player is X or not
        """
        index = self.board.calculate_index_from_position(move[0], move[1])
        matches = self.board.get_neighbours(index, is_player_x)
        changed_chains_index_direction = []
        changed_chains_index_direction.clear()
        if len(matches) == 0:
            self.add_new_chain({index}, is_player_x)
        else:
            for neighbour in matches:
                changed_chains_index_direction.extend(self.add_index_to_chain(index, neighbour, is_player_x))
        if len(changed_chains_index_direction) > 1:
            self.check_for_overlap(changed_chains_index_direction, is_player_x)
        opponent = not is_player_x
        opponent_matches = self.board.get_neighbours(index, opponent)
        for neighbour in opponent_matches:
            self.vet_closed_chains(index, neighbour, opponent)

    def vet_closed_chains(self, index, neighbour, is_opponent_x):
        """
        Check if any of the opponents chains were closed by the last move, remove closed chains from the cache
        :param index: index of the last move
        :param neighbour: neighbouring index of the last move
        :param is_opponent_x: boolean indicating whether the opponent is X
        """
        deletable_indexes = []
        # Check the opponents chains
        for i, index_chain in enumerate(self.x_index_chains) if is_opponent_x else enumerate(self.o_index_chains):
            if neighbour not in index_chain:
                continue
            if len(index_chain) == 1:
                # delete 1 long chain if blocked from all sides
                neighbours_of_neighbour = self.board.calculate_true_neighbouring_indexes(
                    neighbour)  # all possible neighbours of neighbour
                neighbour_count = len(neighbours_of_neighbour)
                if neighbour_count == len(self.board.o_indexes.intersection(
                        neighbours_of_neighbour) if is_opponent_x else self.board.x_indexes.intersection(
                    neighbours_of_neighbour)):
                    deletable_indexes.append(i)
                continue
            chain = sorted(index_chain)
            chain_direction = self.calculate_direction_of_neighbours(chain[0], chain[1])
            negative_closing_index = chain[0] - chain_direction
            positive_closing_index = chain[-1] + chain_direction

            if negative_closing_index != index and positive_closing_index != index:
                continue

            negative_match = negative_closing_index == index
            positive_match = positive_closing_index == index
            negative_in_chain = negative_closing_index in self.board.x_indexes if not is_opponent_x else negative_closing_index in self.board.o_indexes
            positive_in_chain = positive_closing_index in self.board.x_indexes if not is_opponent_x else positive_closing_index in self.board.o_indexes
            blocked_by_edge = self.is_chain_blocked_by_edge(chain_direction, chain[0], chain[-1])
            positive_closing = positive_in_chain or blocked_by_edge
            negative_closing = negative_in_chain or blocked_by_edge

            if (negative_match and positive_closing) or (positive_match and negative_closing):
                deletable_indexes.append(i)

        if len(deletable_indexes) > 0:
            for one_delete in deletable_indexes:
                if is_opponent_x:
                    try:
                        if len(self.x_index_chains[one_delete]) == 4:
                            self.four_x_count += 1
                    except IndexError:
                        pass
                    try:
                        if len(self.x_index_chains[one_delete]) == 3:
                            self.three_x_count += 1
                    except IndexError:
                        pass
                else:
                    try:
                        if len(self.o_index_chains[one_delete]) == 4:
                            self.four_o_count += 1
                    except IndexError:
                        pass
                    try:
                        if len(self.o_index_chains[one_delete]) == 3:
                            self.three_o_count += 1
                    except IndexError:
                        pass
            self.delete_chain_by_index(deletable_indexes, is_opponent_x)

    def is_chain_blocked_by_edge(self, direction, chain_neg_index, chain_pos_index):
        """
        Check if a chain is blocked by edge
        :param direction: direction of the chain
        :param chain_neg_index: the first index of the chain
        :param chain_pos_index: the last index of the chain
        :return: boolean indicating if the chain is blocked by edge
        """
        return (self.is_chain_blocked_top(direction, chain_neg_index)
                or self.is_chain_blocked_left(direction, chain_neg_index, chain_pos_index))

    def is_chain_blocked_left(self, direction, chain_neg_index, chain_pos_index):
        """
        Check if a chain is blocked by edge
        :param direction: direction of the chain
        :param chain_neg_index: the first index of the chain
        :param chain_pos_index: the last index of the chain
        :return: boolean indicating if the chain is blocked by edge
        """
        neg_in_col1 = self.is_index_in_col1(chain_neg_index)
        pos_in_col1 = self.is_index_in_col1(chain_pos_index)
        if ((direction == 1 and neg_in_col1)
                or (direction == self.board.size - 1 and pos_in_col1)
                or (direction == self.board.size + 1 and neg_in_col1)):
            return True
        return False

    def is_chain_blocked_top(self, direction, chain_neg_index):
        """
        Check if a chain is blocked by edge
        :param direction: direction of the chain
        :param chain_neg_index: the first index of the chain
        :return: boolean indicating if the chain is blocked by edge
        """
        neg_in_row1 = self.is_index_in_row1(chain_neg_index)
        if ((direction == self.board.size and neg_in_row1)
                or (direction == self.board.size - 1 and neg_in_row1)
                or (direction == self.board.size + 1 and neg_in_row1)):
            return True
        return False

    def is_neg_blocked(self, direction, chain_neg_index):
        """
        Check if a chain is blocked by edge
        :param direction: direction of the chain
        :param chain_neg_index: the first index of the chain
        :return: boolean indicating if the chain is blocked by edge
        """
        neg_in_row1 = self.is_index_in_row1(chain_neg_index)
        neg_in_col1 = self.is_index_in_col1(chain_neg_index)
        if ((direction == 1 and neg_in_col1)
                or (direction == self.board.size and neg_in_row1)
                or (direction == self.board.size - 1 and neg_in_row1)
                or (direction == self.board.size + 1 and neg_in_row1)):
            return True
        return False

    def is_pos_blocked(self, direction, chain_pos_index):
        """
        Check if positive index is blocked
        :param direction: direction of the chain
        :param chain_pos_index: the last index of the chain
        :return: boolean indicating if the chain is blocked by edge
        """
        pos_in_col1 = self.is_index_in_col1(chain_pos_index)
        if direction == self.board.size - 1 and pos_in_col1:
            return True
        return False

    def is_index_in_row1(self, index):
        """
        Check if the index is in the first row
        :param index: index of a move
        :return: boolean indicating if the index is in the first row
        """
        return index < self.board.size

    def is_index_in_col1(self, index):
        """
        Check if the index is in the first column
        :param index: index of a move
        :return: boolean indicating if the index is in the first column
        """
        return index % self.board.size == 0

    def add_index_to_chain(self, index, neighbour, is_player_x):
        """
        Add an index of a move to chains that contain neighbour
        :param index: index of the move to be added
        :param neighbour: index neighbouring the index parameter
        :param is_player_x: boolean indicating whether the player is X or not
        """
        direction = self.calculate_direction_of_neighbours(index, neighbour)
        changed_chains_index_direction = []
        chains_to_be_added = []
        neighbour_is_dead = True
        for i, index_chain in enumerate(self.x_index_chains) if is_player_x else enumerate(self.o_index_chains):
            if neighbour not in index_chain:
                continue
            neighbour_is_dead = False
            if len(index_chain) == 1:
                index_chain.add(index)
                changed_chains_index_direction.append((i, direction))
                continue
            sorted_chain = sorted(index_chain)
            chain_direction = self.calculate_direction_of_neighbours(sorted_chain[0], sorted_chain[1])
            if direction == chain_direction:
                index_chain.add(index)
                changed_chains_index_direction.append((i, direction))
            else:
                # Create a new chain and add it to the list, if we form a new chain
                # with an index, from all already existing chain
                chains_to_be_added.append(({index, neighbour}, self.calculate_direction_of_neighbours(index, neighbour)))
        if neighbour_is_dead:
            self.add_new_chain({index, neighbour}, is_player_x)
        index_offset = 0
        for chain, direction in chains_to_be_added:
            chain_index = len(self.x_index_chains if is_player_x else self.o_index_chains) + index_offset
            changed_chains_index_direction.append((chain_index, direction))
            self.add_new_chain(chain, is_player_x)
            index_offset += 1
        return changed_chains_index_direction

    def check_for_overlap(self, changed_chains, is_player_x):
        """
        Checks the chains that were chained by the last move in case they overlap
        in case of overlap the function merges the two chains into one
        :param changed_chains: tuple containing the index of the chain in the list of chains
                                and the direction of the chain
        :param is_player_x: boolean indicating whether the player is X or not
        """
        removable_chains = []
        # this treats the symptom but the root cause is still there
        new_changed_chains = []
        for i in range(0, len(changed_chains)):
            if changed_chains[i][0] < len(self.x_index_chains if is_player_x else self.o_index_chains):
                new_changed_chains.append(changed_chains[i])
        changed_chains = copy.deepcopy(new_changed_chains)
        # deep copy to help debugging if needed
        while len(changed_chains) != 0:
            chain_index, chain_direction = changed_chains.pop()
            for index, direction in changed_chains:
                if direction == chain_direction:
                    # merge_chains returns True if the new chain is blocked from both sides, and is safe to remove
                    if self.merge_chains(chain_index, index, is_player_x):
                        removable_chains.append(chain_index)
                    removable_chains.append(index)
        self.delete_chain_by_index(removable_chains, is_player_x)

    def merge_chains(self, index_to_merge_to, index_to_merge, is_player_x):
        """
        Merge to index chains together by their indexes in the list of chains
        :param index_to_merge_to: index in the list of the chain to merge into
        :param index_to_merge: index in the list of the chain to merge
        :param is_player_x: boolean indicating whether the player is X or not
        :return: boolean indicating whether the chain is blocked after the merge or not
        """
        if is_player_x:
            self.x_index_chains[index_to_merge_to].update(self.x_index_chains[index_to_merge])
        else:
            self.o_index_chains[index_to_merge_to].update(self.o_index_chains[index_to_merge])
        return self.is_merged_chain_blocked(index_to_merge_to, is_player_x)

    def is_merged_chain_blocked(self, index_of_chain, is_player_x):
        """
        Check if a new chain is blocked after the merge of two overlapping chains
        :param index_of_chain: the index of the chain in the list of chains
        :param is_player_x: boolean indicating whether the player is X
        :return: boolean indicating whether the chain is blocked after the merge or not
        """
        chain_list = sorted(self.x_index_chains[index_of_chain] if is_player_x else self.o_index_chains[index_of_chain])
        direction = self.calculate_direction_of_neighbours(chain_list[0], chain_list[1])
        blocked_by_edge = self.is_chain_blocked_by_edge(direction, chain_list[0], chain_list[-1])
        pos_blocked = self.is_pos_blocked(direction, chain_list[-1])
        neg_blocked = self.is_neg_blocked(direction, chain_list[0])
        neg_closing_index = chain_list[0] - direction
        pos_closing_index = chain_list[-1] + direction
        neg_occupied = self.board.is_index_occupied(neg_closing_index)
        pos_occupied = self.board.is_index_occupied(pos_closing_index)
        if (neg_occupied or neg_blocked) and (pos_occupied or pos_blocked):
            return True
        return False

    def add_new_chain(self, chain, is_player_x):
        """
        Add new chain to the list of chains for player
        :param chain: chain to be added to list
        :param is_player_x: boolean indicating whether the player is X or not
        """
        if is_player_x:
            self.x_index_chains.append(chain)
        else:
            self.o_index_chains.append(chain)

    def delete_chain_by_index(self, indexes, is_player_x):
        """
        Delete chains from players index chains by their indexes
        :param indexes: list of indexes of the chains to be deleted
        :param is_player_x: boolean indicating whether the player is X or not
        """
        # Reverse sort the indexes, so we don't have to shift them
        # turning it into a set removes duplicates
        # sorted function turns it into a list
        indexes = sorted(set(indexes), reverse=True)
        for index in indexes:
            if is_player_x:
                del self.x_index_chains[index]
            else:
                del self.o_index_chains[index]

    def calculate_direction_of_neighbours(self, index, neighbour):
        """
        Calculate the direction of a chain from neighbouring indexes
        :param index: index of a move
        :param neighbour: index of a move neighbouring the index parameter
        :return: returns direction
        """
        return abs(index - neighbour)

    def check_for_open_chains(self, length, is_player_x):
        """
        Checks is there is a chain with desired length for the player, return the first one
        :param length: length of the chain
        :param is_player_x: boolean indicating whether the player is X
        :return: index of the chain in the list, or None if there is no chain with desired length
        """
        if is_player_x:
            for i, chain in enumerate(self.x_index_chains):
                if len(chain) == length:
                    return i
        else:
            for i, chain in enumerate(self.o_index_chains):
                if len(chain) == length:
                    return i
        return None

    def get_all_open_chains(self, length, is_player_x):
        """
        Checks is there is a chain with desired length for the player, returns all of them
        :param length: length of the chain
        :param is_player_x: boolean indicating whether the player is X
        :return: index of all chains, or None if there is no chain with desired length
        """
        indexes = set()
        if is_player_x:
            for i, chain in enumerate(self.x_index_chains):
                if len(chain) == length:
                    indexes.add(i)
        else:
            for i, chain in enumerate(self.o_index_chains):
                if len(chain) == length:
                    indexes.add(i)
        return indexes

    def find_double_open_3_chains(self, is_player_x):
        """
        Checks if there is a chain with 3 length for the player, open from both sides
        :param is_player_x: indicating whether the player is X or not
        :return: returns all the moves around them
        """
        indexes_of_chains = self.get_all_open_chains(3, is_player_x)
        indexes = set()
        for index in indexes_of_chains:
            chain = sorted(self.x_index_chains[index] if is_player_x else self.o_index_chains[index])
            direction = self.calculate_direction_of_neighbours(chain[0], chain[1])
            negative_closing_index = chain[0] - direction
            positive_closing_index = chain[-1] + direction
            negative_closing_move = self.board.calculate_position_from_index(negative_closing_index)
            positive_closing_move = self.board.calculate_position_from_index(positive_closing_index)
            blocked_by_edge = self.is_chain_blocked_by_edge(direction, chain[0], chain[-1])
            if not blocked_by_edge:
                if (self.board.is_position_valid_from_pos(negative_closing_move[0], negative_closing_move[1]) and
                        self.board.is_position_valid_from_pos(positive_closing_move[0], positive_closing_move[1])):
                    indexes.add(negative_closing_index)
                    indexes.add(positive_closing_index)
        return indexes

    def get_all_chain_edge_indexes(self, lenght, is_player_x):
        """
        Checks is there is a chain with desired length for the
        player, return all the indexes
        :param lenght: length of the chain, must be at least 2
        :param is_player_x: boolean indicating whether the player is
        """
        indexes_of_chains = self.get_all_open_chains(lenght, is_player_x)
        indexes = set()
        for index in indexes_of_chains:
            chain = sorted(self.x_index_chains[index] if is_player_x else self.o_index_chains[index])
            direction = self.calculate_direction_of_neighbours(chain[0], chain[1])
            negative_closing_index = chain[0] - direction
            positive_closing_index = chain[-1] + direction
            negative_closing_move = self.board.calculate_position_from_index(negative_closing_index)
            positive_closing_move = self.board.calculate_position_from_index(positive_closing_index)
            blocked_by_edge = self.is_chain_blocked_by_edge(direction, chain[0], chain[-1])
            if not blocked_by_edge:
                if self.board.is_position_valid_from_pos(negative_closing_move[0], negative_closing_move[1]):
                    indexes.add(negative_closing_index)
                elif self.board.is_position_valid_from_pos(positive_closing_move[0], positive_closing_move[1]):
                    indexes.add(positive_closing_index)
            else:
                if direction == 1 or direction == self.board.size or direction == self.board.size + 1 or (
                        direction == self.board.size - 1 and self.is_index_in_row1(negative_closing_index)):
                    indexes.add(positive_closing_index)
                else:
                    indexes.add(negative_closing_index)
        return indexes

    def get_available_moves_around_1_long_chains(self):
        """
        Calculates all the available moves around 1 long chains
        :return: list of all available moves around 1 long chains, returns an empty list if there is none
        """
        one_longs = []
        moves = set()
        for chain in self.x_index_chains:
            if len(chain) == 1:
                one_longs.append(list(chain)[0])
        for chain in self.o_index_chains:
            if len(chain) == 1:
                one_longs.append(list(chain)[0])
        if len(one_longs) > 0:
            for index in one_longs:
                moves = moves.union(self.board.calculate_true_neighbouring_indexes(index))
            moves = (moves - self.board.x_indexes) - self.board.o_indexes
        return list(moves)

    def check_for_4_move(self, is_player_x):
        # I'm a very optimistic function, I assume that there are no bugs are happening before running me
        """
        Check if the bot has a 4 long chain to win
        or check if the opponent has a 4 long chain that the bot has to block
        :param is_player_x: boolean indicating whether the player is X
        :return: coordinates of the move or None
        """
        index_of_chain = self.check_for_open_chains(4, is_player_x)
        if index_of_chain is not None:
            # Bot can win with 4 long chain
            # Player can win with 4 win chain, bot has to block it
            chain = sorted(self.x_index_chains[index_of_chain] if is_player_x else self.o_index_chains[index_of_chain])
            direction = self.calculate_direction_of_neighbours(chain[0], chain[1])
            negative_closing_index = chain[0] - direction
            positive_closing_index = chain[-1] + direction
            negative_closing_move = self.board.calculate_position_from_index(negative_closing_index)
            positive_closing_move = self.board.calculate_position_from_index(positive_closing_index)
            blocked_by_edge = self.is_chain_blocked_by_edge(direction, chain[0], chain[-1])
            if not blocked_by_edge:
                if self.board.is_position_valid_from_pos(negative_closing_move[0], negative_closing_move[1]):
                    return negative_closing_move
                elif self.board.is_position_valid_from_pos(positive_closing_move[0], positive_closing_move[1]):
                    return positive_closing_move
            else:
                if direction == 1 or direction == self.board.size or direction == self.board.size + 1 or (
                        direction == self.board.size - 1 and self.is_index_in_row1(negative_closing_index)):
                    return positive_closing_move
                else:
                    return negative_closing_move
        return None

    # TODO: this is the old function as it was
    def check_for_4_move_old(self, is_player_x):
        """
                Check if the bot has a 4 long chain to win
                or check if the opponent has a 4 long chain that the bot has to block
                :param is_player_x: boolean indicating whether the player is X
                :return: coordinates of the move or None
                """
        index_of_chain = self.check_for_open_chains(4, is_player_x)
        move = None
        if index_of_chain is not None:
            # Bot can win with 4 long chain
            # Player can win with 4 win chain, bot has to block it
            chain = sorted(self.x_index_chains[index_of_chain] if is_player_x else self.o_index_chains[index_of_chain])
            direction = self.calculate_direction_of_neighbours(chain[0], chain[1])
            negative_closing_index = chain[0] - direction
            positive_closing_index = chain[-1] + direction
            blocked_by_edge = self.is_chain_blocked_by_edge(direction, chain[0], chain[-1])
            if not blocked_by_edge:
                if self.board.is_index_occupied(negative_closing_index):
                    move = self.board.calculate_position_from_index(positive_closing_index)
                    if self.board.is_index_occupied(positive_closing_index):
                        raise RuntimeError
                else:
                    move = self.board.calculate_position_from_index(negative_closing_index)
            else:
                if direction == 1:
                    # With horizontal direction this is the only possible move,
                    # positive_closing_index should be free, otherwise it would have been filtered out previously
                    # by vetting the closed chains
                    # TODO: new index off the edge check
                    move = self.board.calculate_position_from_index(positive_closing_index)
                    if self.board.is_index_occupied(positive_closing_index):
                        print("There is a bug in check_for_4_move functions horizontal move searching")
                        raise RuntimeError
                elif direction == self.board.size - 1:
                    # With vertical direction this is the only possible move,
                    # positive_closing_index should be free, otherwise it would have been filtered out previously
                    # by vetting the closed chains
                    # TODO: new index off the edge check
                    if self.is_index_in_row1(negative_closing_index):
                        move = self.board.calculate_position_from_index(positive_closing_index)
                        if self.board.is_index_occupied(positive_closing_index):
                            print("There is a bug in check_for_4_move functions diagonal down-up move searching")
                            raise RuntimeError
                    else:
                        move = self.board.calculate_position_from_index(negative_closing_index)
                        if self.board.is_index_occupied(negative_closing_index):
                            print("There is a bug in check_for_4_move functions diagonal down-up move searching")
                            raise RuntimeError
                elif direction == self.board.size:
                    # With vertical direction this is the only possible move,
                    # positive_closing_index should be free, otherwise it would have been filtered out previously
                    # by vetting the closed chains
                    # TODO: new index off the edge check
                    move = self.board.calculate_position_from_index(positive_closing_index)
                    if self.board.is_index_occupied(positive_closing_index):
                        print("There is a bug in check_for_4_move functions vertical move searching")
                        raise RuntimeError
                elif direction == self.board.size + 1:
                    # With diagonal up-down direction this is the only possible move,
                    # positive_closing_index should be free, otherwise it would have been filtered out previously
                    # by vetting the closed chains
                    # TODO: new index off the edge check
                    move = self.board.calculate_position_from_index(positive_closing_index)
                    if self.board.is_index_occupied(positive_closing_index):
                        print("There is a bug in check_for_4_move functions diagonal up-down move searching")
                        raise RuntimeError
        return move

    def smart_move(self, last_move, enlarged):
        if enlarged:
            self.recalculate_chains()
        self.add_last_move(last_move, True)
        # Check for win condition
        move = self.check_for_4_move(False)
        if move is not None:
            self.add_last_move(move, False)
            return move
            # Check for opponent win condition to block
        move = self.check_for_4_move(True)
        if move is not None:
            self.add_last_move(move, False)
            return move
        head = Node(False, bot=self)
        head.set_value(self.minimax(head, 0, True, float("-inf"), float("inf")))
        step_tuple = self.board.calculate_position_from_index(self.step)
        self.add_last_move(step_tuple, False)
        return step_tuple
