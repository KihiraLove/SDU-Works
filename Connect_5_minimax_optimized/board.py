class Board:
    def __init__(self):
        """
        Define a board for the 5 steps game
        Starting size is 20
        """
        self.size = 20
        self.x_indexes = set()
        self.o_indexes = set()

    def enlarge(self):
        """
        Enlarges the board by 1 x and 1 y
        Recalculates the index sets for each player
        """
        self.x_indexes = self.shift_indexes(self.x_indexes)
        self.o_indexes = self.shift_indexes(self.o_indexes)
        self.size += 1

    def shift_indexes(self, indexes):
        """
        Recalculates an index set for enlargement
        Each index has to be shifted to the right by the amount of new indexes inserted by the enlargement.
        There is one new index inserted to the end of each row.
        We are adding the index of the row to all the indexes in the row, for each row
        :param indexes: the indexes to be shifted
        :return: new shifted indexes
        """
        new_indexes = set()
        for index in indexes:
            # index // self.size returns the row index
            new_index = index + index // self.size
            new_indexes.add(new_index)
        return new_indexes

    def get_all_possible_moves_in_range(self, move_range=1):
        if move_range > 2 or move_range < 1:
            raise RuntimeError
        all_moves = {}
        for index in self.x_indexes:
            all_moves = all_moves.union(self.calculate_true_neighbouring_indexes(index))
        for index in self.o_indexes:
            all_moves = all_moves.union(self.calculate_true_neighbouring_indexes(index))
        if move_range == 1:
            all_moves = (all_moves - self.x_indexes) - self.o_indexes
            return all_moves
        working_set = all_moves.copy()
        for index in all_moves:
            working_set = working_set.union(self.calculate_true_neighbouring_indexes(index))
        working_set = (working_set - self.x_indexes) - self.o_indexes
        return working_set

    def calculate_index_from_position(self, x, y):
        """
        Calculates the index of the move from board position
        :param x: row number
        :param y: column number
        :return: calculated index
        """
        return (x - 1) * self.size + y - 1

    def calculate_position_from_index(self, index):
        """
        Calculates the position of the move from index
        :param index: the index of the move
        :return: the row and column number of the move
        """
        row_index = index // self.size
        column_index = index - row_index * self.size
        return (row_index + 1), (column_index + 1)

    def add_index(self, index, is_player_x):
        """
        Adds the index of the move to the respective players indexes
        :param index: index of the move
        :param is_player_x: boolean indicating whether the player is X or not
        """
        if is_player_x:
            self.x_indexes.add(index)
        else:
            self.o_indexes.add(index)

    def set_position(self, x, y, is_player_x):
        """
        Sets the position of the move for the respective player
        Enlarges the board by 1 x and 1 y if the move is out of bounds
        :param x: row number
        :param y: column number
        :param is_player_x: boolean indicating whether the player is X or not
        :return: boolean indicating if the board was enlarged by the move or not
        """
        is_enlarged = False
        if x > self.size or y > self.size:
            self.enlarge()
            is_enlarged = True
        self.add_index(self.calculate_index_from_position(x, y), is_player_x)
        return is_enlarged

    def is_position_valid_from_pos(self, x, y):
        """
        Checks if the move is valid
        :param x: row number
        :param y: column number
        :return: boolean indicating if the move is valid
        """
        index = self.calculate_index_from_position(x, y)
        # Move is invalid if the index of the move is already occupied or the move is out of bounds by more than 1
        if self.is_index_occupied(index) or x > self.size + 1 or y > self.size + 1 or x < 1 or y < 1:
            return False
        return True

    def is_index_occupied(self, index):
        """
        Checks if the index is occupied by either of the players
        :param index: index to be checked
        :return: boolean indicating weather the index is occupied or not
        """
        return self.is_index_in_indexes_for_player(index, True) or self.is_index_in_indexes_for_player(index, False)

    def is_index_in_indexes_for_player(self, index, is_player_x):
        """
        Checks is the index is in the index set of the player
        :param index: index to be checked
        :param is_player_x: boolean indicating whether the player is X or not
        :return: boolean indicating if the index is in the players index set
        """
        if is_player_x:
            return index in self.x_indexes
        else:
            return index in self.o_indexes

    def get_neighbours(self, index, is_player_x):
        """
        Returns the neighbours of the index for the current player
        :param index: index to be checked
        :param is_player_x: boolean indicating whether the player is X or not
        :return: all the neighbours of the index for the current player
        """
        neighbours = self.calculate_true_neighbouring_indexes(index)
        # Check if the valid neighbours of the last move are among the current players move indexes
        return self.x_indexes.intersection(neighbours) if is_player_x else self.o_indexes.intersection(neighbours)

    def check_for_win(self, x, y, is_player_x, chain_length=5):
        """
        Checks if the last move won the game
        :param x: row number
        :param y: column number
        :param is_player_x: boolean indicating whether the player is X or not
        :param chain_length: is the length of the chain required to win the game
        :return: boolean indication if the last move won the game
        """
        index = self.calculate_index_from_position(x, y)
        matches = self.get_neighbours(index, is_player_x)
        horizontal_direction = 1
        vertical_direction = self.size
        diagonal_up_down = self.size + 1
        diagonal_down_up = self.size - 1
        if len(matches) == 0:
            return False
        # Horizontal connection between neighbours and last move
        if (index - horizontal_direction) in matches or (index + horizontal_direction) in matches:
            if self.check_for_chain(chain_length, horizontal_direction, index, is_player_x):
                return True
        # Vertical connection between neighbours and last move
        elif (index - vertical_direction) in matches or (index + vertical_direction) in matches:
            if self.check_for_chain(chain_length, vertical_direction, index, is_player_x):
                return True
        # Left upper right lower connection between neighbours and last move
        elif (index - diagonal_up_down) in matches or (index + diagonal_up_down) in matches:
            if self.check_for_chain(chain_length, diagonal_up_down, index, is_player_x):
                return True
        # Left lower right upper connection between neighbours and last move
        elif (index - diagonal_down_up) in matches or (index + diagonal_down_up) in matches:
            if self.check_for_chain(chain_length, diagonal_down_up, index, is_player_x):
                return True
        return False

    def check_for_chain(self, chain_length, direction, index, is_player_x):
        """
        Checks if there is a chain of past moves with the last move for the current player that can lead to a win
        :param chain_length: the length of the searched chain
        :param direction: direction of the chain
        :param index: index of the last move
        :param is_player_x: boolean indicating whether the player is X or not
        :return: boolean indicating that there is a chain of moves for the current player
        """
        count = 1
        started = index
        checked = index
        direction_is_positive = False
        directions_checked = 0
        while True:
            # Return True if the chain is at least 5 long
            if count == chain_length:
                return True
            # Break and return False if we checked in both directions and the chain is less than 5 long
            if directions_checked == 2:
                break
            # Calculate the neighbour in the currently checked direction
            neighbour = checked + direction if direction_is_positive else checked - direction
            # If the neighbour is invalid or not contained in the current players past move indexes reverse direction
            # and start checking from the original index
            if self.neighbour_breaks_rule(neighbour, checked) or (
                    (neighbour not in self.x_indexes and is_player_x) or (
                    neighbour not in self.o_indexes and not is_player_x)):
                direction_is_positive = not direction_is_positive
                checked = started
                directions_checked += 1
            # If the neighbour is valid and is contained in the current players past moves,
            # checks its neighbour in the current direction
            else:
                count += 1
                checked = neighbour
        return False

    def calculate_true_neighbouring_indexes(self, index):
        """
        Calculate the neighbours of the current index
        :param index: current index
        :return: set containing all the valid neighbours
        """
        neighbours = self.calculate_neighboring_indexes(index)
        # Remove all invalid neighbours
        neighbours = self.vet_neighbouring_indexes(neighbours, index)
        return neighbours

    def calculate_neighboring_indexes(self, index):
        """
        Calculate the mathematical neighbours of the current index
        These indexes might not be true neighbours,
        if the index is on the left edge, its mathematical neighbours will be on the right edge,
        and not be a true neighbour our
        :param index: index to be checked
        :return: set containing all the indexes that are mathematically neighbours of the current index
        """
        # Add all neighbours into a set
        mathematical_neighbours = {index - self.size - 1, index - self.size, index - self.size + 1,
                                    index - 1, index + 1,
                                    index + self.size - 1, index + self.size, index + self.size + 1}
        return sorted(mathematical_neighbours)

    def vet_neighbouring_indexes(self, neighbours, index):
        """
        Remove all invalid neighbours of the current index from the neighbours set
        :param neighbours: all the neighbours of the current index
        :param index: current index
        :return: set of the valid neighbours of the current index
        """
        working_set = neighbours.copy()
        for neighbour in neighbours:
            # Remove the neighbour if it breaks any rules
            if self.neighbour_breaks_rule(neighbour, index):
                working_set.remove(neighbour)
        return working_set

    def neighbour_breaks_rule(self, neighbour, index):
        """
        Checks if the neighbour breaks any rules
        :param neighbour: the index we are checking
        :param index: the index we are checking against
        :return: Boolean indicating if the neighbour breaks any rules
        """
        # If the neighbour index is negative, the position is not on the board
        # If the neighbour index is larger or equals to self.size * self.size, the position is not on the board
        # If the index is on the left edge and the neighbour is on the right edge, they are not true neighbours
        # If the index is on the right edge and the neighbour is on the left edge, they are not true neighbours
        return (neighbour < 0
                or neighbour >= self.size * self.size
                or (index % self.size == 0 and neighbour % self.size == self.size - 1)
                or (index % self.size == self.size - 1 and neighbour % self.size == 0))

    def move(self, x, y, is_player_x):
        """
        Set the next move on the board
        :param x: row number
        :param y: column number
        :param is_player_x: boolean indicating whether the player is X or not
        :raises IndexError: if the move is invalid
        :return: is_win, is_enlarged: boolean indicating if the move won the game,
                                     and boolean indicating if the board was enlarged by the move or not
        """
        if not self.is_position_valid_from_pos(x, y):
            raise IndexError
        is_enlarged = self.set_position(x, y, is_player_x)
        is_win = self.check_for_win(x, y, is_player_x)
        return is_win, is_enlarged

    def print_board(self):
        """
        Prints the current board state
        """
        # Construct the header of the board
        head_string = "    " + "".join(f"{i+1}   " if i < 9 else f"{i+1}  " for i in range(self.size))
        # Construct the board state
        board_string = "".join(
            " X |" if i in self.x_indexes else
            " O |" if i in self.o_indexes else
            "   |" for i in range(self.size * self.size)
        )
        index_shift = 0
        number_of_chars_in_cell = 4
        # Insert row labels into the board state
        for i in range(self.size):
            string_index = (i * self.size) * number_of_chars_in_cell + index_shift
            board_string = board_string[:string_index] + (f"\n{i + 1} |" if i < 9 else f"\n{i + 1}|") + board_string[string_index:]
            index_shift += 4

        print(head_string + board_string)
