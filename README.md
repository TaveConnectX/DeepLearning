# Connect X

Connect 4 agent with Reinforcement Learning
Connect4 게임을 강화학습을 이용해서 학습시키는 프로젝트 
백엔드 및 안드로이드와 결합해서 어플로 구현시키는 것이 목표이다. 

* agent_structure.py: 여러 agent의 구조 클래스 파일
* config.json: 모델을 학습할 때 변경할 파라미터 파일
* env.py: Connect 4 게임 환경을 구현한 클래스
* evaluate_alphazero.py: 알파제로 및 알파고의 성능 측정을 위한 파일
* evaluate_model.py: 그 외 모델의 성능 측정을 위한 파일
* find_play_param_alphazero.py: 알파제로의 C, num_search 의 하이퍼파라미터 샘플링을 위한 파일
* functions.py: 여러 함수들을 모아놓은 파일
* model_comparison.py: 두 모델의 성능을 비교하기 위한 파일
* models.py: 모델의 구조를 나타내는 파일
* play_game.py 주어진 모델로 Connect 4를 플레이할 수 있는 파일
* ReplayBuffer.py: DQN의 리플레이 버퍼를 이용하기 위한 클래스 파일
* requirement.txt: 파일 실행을 위한 모듈 모음
* test_alphago.py: 알파고 모델의 작동 테스트를 확인하기 위한 파일
* test_alpha_model.py: 알파제로 모델의 작동 테스트를 확인하기 위한 파일
* test_model.py: 그 외 모델의 작동 테스트를 확인하기 위한 파일
* test_value_net.py: 알파고에 사용되는 value network 모델의 작동 테스트를 확인하기 위한 파일
* train_AZ.py: 알파제로를 학습하는 파일
* train.py: 그 외 모델을 학습하는 파일
* train_PS.py: DQN 계열 모델의 하이퍼파라미터 샘플링을 하기 위한 파일
* train_ValueNetwork.py: 알파고에 사용되는 Value network를 학습하는 파일


* alphazero_new.py: agent_structure.py와 병합
* AlphaZeroenv.py: env.py 와 병합
* for_DJ.py: train_value_net.py 와 병합
* json2excel.py: train_PS.py 와 병합
* selfplay.py: train.py와 병합