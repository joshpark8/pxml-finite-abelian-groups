# Combinatorics of Finite Abelian Group Actions

This repository contains computational tools from a Spring 2026 Purdue Experimental Mathematics Lab project on actions of finite abelian groups on finite sets.

Team: Josh Park, Arthur Illidge, Carson Gull
Project Advisors: Prof. Shaver Phagan, Benjamin Mudrak
Program Supervisor: Prof. Thomas Sinclair

## Our question

Let $G$ be a finite abelian group of order $n$ acting on a finite set $X$. If $M_H$ is the number of orbits induced by the $H$-action on $X$ and $a_d$ is the number of orbits with cardinality $d$ in $X$, we ask: is knowledge of the families $(M_H)_{H\leqslant G}$ and $(a_d)_{d\mid n}$ sufficient to characterize $X$ up to isomorphism?

The answer: not in general. Interested readers can read our [preprint](https://arxiv.org/abs/2610.16880) for the proof.

## Purpose

Scripts in this repository served many purposes throughout our research, but were primarily used for counting orbits of finite abelian $G$-sets and testing various conjectures during the project.

## Status

This repository was created explicitly for our internal experimental research code and is made public for the sake of transparency and in case anyone doing parallel research might find it useful. However, we stress that everything was designed for internal use; hence we do not claim the accuracy of any code or mathematics in this repository. See our [preprint](https://arxiv.org/abs/2610.16880) for the formal mathematics.

## Authorship

While assistance from large language models (specifically, Anthropic's Claude Opus) was used in the development of these tools, all code was written under direct human supervision and outputs were verified by the team. All mathematical questions and conjectures being tested were from the team.
