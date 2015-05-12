from __future__ import division
import sys, math, operator, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class Grid():

	def __init__(self):
		self.states = []
		self.rewards = {}
		self.utility = {}
		self.actions = {}
		self.action_move()
		self.iterations = []
		self.gamma = 0.99
		self.policy = {}
		# number of times we've taken action from state s
		self.n_s_a = {}
		# number of trials
		self.trial = []
		self.rmse = []
		self.maxtrial = 1000
		self.maxstep = 50
		# cutoff for exploration
		self.Ne = 20
		# q values
		self.q = {}
		self.expectedUtil = {}

	def learning_rate(self, t):
		return 60 / (59 + t)

	def explore_func(self, u, n):
		if n < self.Ne:
			return sys.maxint
		else:
			return u

	def optimal_action(self, state):
		"""determine the optimal action to make"""
		d = {}
		for action in self.actions:
			s_a = (state, action)
			d[action] = self.explore_func(self.q.get(s_a,0), self.n_s_a.get(s_a,0))
		return max(d.iteritems(), key=operator.itemgetter(1))[0]

	def max_q(self, state):
		m = []
		for action in self.actions:
			s_a = (state, action)
			m.append(self.q.get(s_a,0))
		return max(m)

	def reinforcement_learning(self):
		for trial in range(1, self.maxtrial):
			self.trial.append(trial)
			#pick a random starting state
			index = random.randint(0,len(self.states)-1)
			state = self.states[index]
			for t in range(1, self.maxstep+1):
				alpha = self.learning_rate(t)
				move = self.optimal_action(state)
				col = state[0]
				row = state[1]
				if move == 'up':
					new_state = (col, row-1)
				elif move == 'left':
					new_state = (col-1, row)
				elif move == 'right':
					new_state = (col+1, row)
				else:
					new_state = (col, row+1)
				if new_state not in self.states:
					new_state = (col, row)
				# update Q
				s_a = (state, move)
				self.n_s_a[s_a] = self.n_s_a.get(s_a,0) + 1
				self.q[s_a] = self.q.get(s_a, 0) + alpha * (self.rewards[state] + self.gamma \
							* self.max_q(new_state) - self.q.get(s_a, 0))
				state = new_state
			for s in self.states:
				if s not in self.expectedUtil:
					self.expectedUtil[s] = []
				self.expectedUtil[s].append(self.max_q(s))
			self.rmse.append(self.cal_rmse())

	def cal_rmse(self):
		sum_diff = 0
		for state in self.states:
			true = self.utility[state][-1]
			expect = self.expectedUtil[state][-1]
			sum_diff += (true - expect)**2
		return math.sqrt(sum_diff / len(self.states))

	def action_move(self):
		"""dictionary for intend move and probabilities"""
		self.actions['up'] = {'up':0.8, 'left':0.1, 'right':0.1}
		self.actions['down'] = {'down':0.8, 'left':0.1, 'right':0.1}
		self.actions['left'] = {'left':0.8, 'up':0.1, 'down':0.1}
		self.actions['right'] = {'right':0.8, 'up':0.1, 'down':0.1}

	def init_grid(self):
		"""initialize the grid with state and rewards"""
		# add states
		for i in range(6):
			for j in range(6):
				self.states.append((i,j))
		# remove walls
		for item in [(1,0),(1,4),(2,4),(3,4),(4,1)]:
			self.states.remove(item)
		for state in self.states:
			# initialize utilities to 0
			self.utility[state] = []
			self.utility[state].append(0)
			# update the rewards
			if state in [(0,0),(2,0),(3,1),(4,2),(5,0),(5,3)]:
				self.rewards[state] = 1
			elif state in [(1,1),(2,2),(3,3),(4,4),(5,1)]:
				self.rewards[state] = -1
			else:
				self.rewards[state] = -0.04

	def expect_utility(self, state):
		"""calculate the expected utility given the actions"""
		col = state[0]
		row = state[1]
		expect_util = {}
		for action in self.actions:
			moves = self.actions[action]
			sum_moves = 0
			for move, prob in moves.iteritems():
				if move == 'up':
					new_state = (col, row-1)
				elif move == 'left':
					new_state = (col-1, row)
				elif move == 'right':
					new_state = (col+1, row)
				else:
					new_state = (col, row+1)

				if new_state not in self.states:
					# we are on the edge or we hit the wall, bounced back to original state
					new_state = (col, row)

				new_state_util = self.utility[new_state][-1]
				sum_moves += prob * new_state_util

			expect_util[action] = sum_moves
		best_policy = max(expect_util.iteritems(), key=operator.itemgetter(1))[0]
		self.policy[state] = best_policy
		return expect_util[best_policy]

	def value_iteration(self, epsilon = 0.001):
		"""perform value iteration"""
		it = 1
		self.iterations.append(0)
		while True:
			delta = 0
			self.iterations.append(it)
			for state in self.states:
				curr_util = self.utility[state][-1]
				new_util = self.rewards[state] + self.gamma * self.expect_utility(state)
				self.utility[state].append(new_util)
				delta = max(delta, abs(curr_util - new_util))
			if delta < epsilon * (1 - self.gamma) / self.gamma:
				return
			#print "iteration: %i" %(it)
			#print "delta is: %g" %(delta)
			it += 1

if __name__ == "__main__":
	grid = Grid()
	grid.init_grid()
	grid.value_iteration()
	print "Utilities of all states:"
	for i in range(6):
		for j in range(6):
			state = (i,j)
			if state in grid.utility.keys():
				print state, ":", grid.utility[state][-1]

	print "-----------------------"
	print "Optimal policy:"
	for i in range(6):
		for j in range(6):
			state = (i,j)
			if state in grid.utility.keys():
				print state, ":", grid.policy[state]

	# plot utility estimates as a function of the number of iterations 
	fontP = FontProperties()
	fontP.set_size('xx-small')
	plt.plot()
	for i in range(6):
		for j in range(6):
			state = (i,j)
			if state in grid.utility.keys():
				plt.plot(grid.iterations, grid.utility[state], label=str(state))
	plt.xlabel('Number of iterations')
	plt.ylabel('Utility estimation')
	plt.legend(prop=fontP, loc='best')
	plt.show()

	grid.reinforcement_learning()

	plt.plot()
	for i in range(6):
		for j in range(6):
			state = (i,j)
			if state in grid.utility.keys():
				plt.plot(grid.trial, grid.expectedUtil[state], label=str(state))
	plt.xlabel('Number of trials')
	plt.ylabel('Utility estimation')
	plt.legend(prop=fontP, loc='best')
	plt.show()

	plt.plot(grid.trial, grid.rmse, label = 'RMSE')
	plt.xlabel('Number of trials')
	plt.ylabel('Root mean square error')
	plt.legend(loc='best')
	plt.show()
