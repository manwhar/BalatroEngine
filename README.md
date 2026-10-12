# BalatroEngine
Finding the optimal move in Balatro.

### Expected workflow:
--> Create an action classifier to scrape pro level YouTube videos for gameplay data
*then* Pretrain a model to take actions given a gamestate under supervised learning
*then* Continue training under reinforcement learning using a headless CLI Balatro clone
*then* Integrate the model as a heuristic for a Monte-Carlo tree search
