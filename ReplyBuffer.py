from collections import deque
import math, random

class ReplayBuffer(object):
	def __init__(self,ReplayMemory):
		
		#self.state_size = state_size
		self.buffer = deque(maxlen=ReplayMemory)

	def push(self, state, M, iteration):

		self.buffer.append(( (state[0].detach(),state[1].detach(),state[2].detach(),state[3].detach()), M.detach(),  iteration.detach()))
		

	def sample(self,batch_size):

		state, M, iteration= zip(*random.sample(self.buffer, batch_size))

		return state, M, iteration
		
	def shuffle(self):
		
		random.shuffle(self.buffer)

	def __len__(self):
		return len(self.buffer)
