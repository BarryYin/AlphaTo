import os

#os.system('pip install -e ./agentscope')
base_path = './internlm2-chat-7b'
os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-7b.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system(f'lmdeploy serve api_server {base_path}  --server-port 23333')
os.system('python app_game_5.py')


