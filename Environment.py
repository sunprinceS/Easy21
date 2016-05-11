"""
File: Environment.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: Easy21 Simulator
"""

import sys
import random

from marcos import *
### Env Marcos
red_prob = 1.0/3
dealer_threshold = 16

def init():
    return (random.randint(0,9),random.randint(0,9))

def check_burst(card_sum):
    return (card_sum>20) or (card_sum<0)

def step(state,action):
    """
    state : tuple(dealer's first card -1, player's current sum -1)
    action : Stick: 0; Hit: 1

    Return:
        next_state : tuple or TERMINAL
        reward : -1, 0, 1
    """
    dealer_sum,player_sum = state

    if action == HIT:
        # print("player's origin sum is {}".format(player_sum))
        player_sum += draw_card()
        # print("player's current sum is {}".format(player_sum))
        if check_burst(player_sum):
            return TERMINAL, -1
        else:
            return (dealer_sum,player_sum), 0
    else: #stick
        dealer_sum = deal_to_dealer(dealer_sum)
        if player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        else:
            reward = 0
        return TERMINAL,reward

def deal_to_dealer(init_sum):
    dealer_sum = init_sum
    while True:
        dealer_sum += draw_card()

        if check_burst(dealer_sum):
            # print("Dealer GG {}".format(dealer_sum))
            return -100  #CON TSAN HAO
        if dealer_sum >= dealer_threshold:
            # print("dealer's final sum is {}".format(dealer_sum))
            return dealer_sum

def draw_card():
    prob = random.uniform(0,1)
    value = random.randint(1,10)
    if prob >= red_prob:
        # print("BLACK {}".format(value))
        return value
    else:
        # print("RED {}".format(value))
        return -1 * value

if __name__ == "__main__":
    dealer_sum,player_sum = init()
    print("dealer's first card is black {}".format(dealer_sum+1))
    print("player's first card is black {}".format(player_sum+1))
    while True:
        action = input("Action? ")
        if action == 'h':
            state,reward = step((dealer_sum,player_sum),HIT)
            print(state)
        else:
            state,reward = step((dealer_sum,player_sum),STICK)
            print(state)

        if state == TERMINAL:
            print("Game over! The reward is {}".format(reward))
            sys.exit(1)
        dealer_sum,player_sum = state
