# Game_gomoku in AgentScope
This is a demo of how to program gomoku_agent in AgentScope.
Complete code is in `game_gomoku.py`
You can start the game like this.
```bash
#Note: if you have not installed agentscope or renew agentscope, you can install it by running the following command
pip install -e .

# Note: Set your api_key in game_gomoku.py first 
YOUR_MODEL_CONFIGURATION_NAME = "gpt-4"
YOUR_MODEL_CONFIGURATION = {
    "config_name": "gpt-4",
    "model_type": "openai",
    "model_name": "gpt-4",
    "api_key": "",  # Load from env if not provided
    "organization": "",  # Load from env if not provided
    "generate_args": {
                    "temperature": 0.5,
                },
    "api_url":""
}

#run the game
python game_gomoku.py
```