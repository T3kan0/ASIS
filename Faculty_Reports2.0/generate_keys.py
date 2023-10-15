#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  19 15:30:00 2021

@author: nt4-nani
"""
import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ['Tekano Mbonani', 'Evodia Mohoanyane']
usernames = ['tmbonani', 'emohoanyane']
passwords = ['abc123', 'def456']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('wb') as file:
    pickle.dump(hashed_passwords, file)














