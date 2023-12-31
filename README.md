# genetic-algorithm_cartrider

인공지능을 유전적 알고리즘을 통해 학습시켜 트랙을 통과할 수 있도록 하는 프로젝트입니다.
각 카트는 고유의 신경망을 가지고 있고, 카트의 전방, +30도, +60도, -30도, -60도 방향으로 벽이 얼마나 멀리 있는지를 확인할 수 있습니다.
이전 세대에서 가장 성과가 좋았던 카트는 사라지지 않고, 다음 세대로 넘어갈때 자손을 남길 확률이 크게 증가합니다.
이전 세대에서 두번째로 성과가 좋았던 카트는 다음세대로 넘어갈때 자손을 남길 확률이 증가합니다.
성과는 트랙을 통과하지 못했을때는 트랙의 진행도에 비례하는 점수를 받고, 트랙을 통과하였을때는 통과한 시간이 짧을수록 더 높은 점수를 받습니다.

현재 구현된 기능은 다음과 같습니다.

 스펙터 모드(디폴트)
이전 세대에서 가장 성과가 좋았던 카트를 따라가면서 진행상황을 볼 수 있게 합니다.
플레이 모드에서 a키를 누르면 스펙터 모드로 전환됩니다.
스펙터 모드에서는 기록 갱신 로그를 한번에 최대 6개까지 볼 수 있습니다. 또한, l키를 눌러 더 많은 로그(최대 25개)를 볼 수 있습니다.

 플레이 모드
스펙터 모드에서 a키를 누르면 플레이 모드로 전환되고, 직접 훈련중인 카트들과 함께 레이스를 진행할 수 있습니다.
조작법은 다음과 같습니다:
위 화살표: 엑셀
왼쪽 화살표: 좌회전
오른쪽 화살표: 우회전
z키: 드리프트
드리프트를 사용하는 중에는 부스터가 켜진 상황에서도 가속되지 않으며, 평소보다 더 급격하게 방향을 꺾을 수 있고, 부스터 게이지가 충전됩니다.
부스터가 켜진 상태에서는 평균적인 속도가 약 1.7초간 100% 상승됩니다.

 키 뷰어
현재 따라가고 있는 카트의 키가 어떻게 입력되고 있는지를 왼쪽 아래에서 확인할 수 있습니다. 각 버튼은 드리프트, 좌회전, 엑셀, 우회전을 나타냅니다.

 세이브/로드
ctrl + s키를 눌러 현재의 진행상황을 저장할 수 있습니다. 저장된 세이브 파일은 새로 프로그램을 실행할때 이어서 진행할지 선택할 수 있습니다.
예시 파일로 성과가 가장 좋았던 19.72초의 기록을 가진 세이브 파일을 저장해두었습니다.

 저의 대학 진학 이후 첫 프로젝트에 관심을 가져주셔서 감사합니다. 게임엔진을 사용하지 않고 순수 파이썬으로 진행한 프로젝트이다보니 유전적 알고리즘보다
학습 환경 및 최적화에 훨씬 많은 시간을 투자한 것 같습니다. 자체엔진이라 버그가 있을 수 있고 효율적이지 않지만 개인적으로 매우 재미있었던 프로젝트였습니다.
앞으로 제가 시간이 날때 한번씩 이러한 짧은 프로젝트를 제작할 예정입니다. 제가 진행할 여러가지 인공지능/게임 등등 여러 프로젝트에도 많은 기대와 응원 부탁드립니다. 감사합니다.

프로젝트에 대한 궁금한점이 있으시다면 아래 연락처로 연락하실 수 있습니다.
제작자 연락처: kevin051211@kaist.ac.kr

(google translated)
It is a project that trains artificial intelligence through genetic algorithms so that it can pass the track.
Each cart has its own neural network, and it can determine how far the wall is in the forward, +30, +60, -30, and -60 degree directions of the cart.
The karts that performed best in the previous generation do not disappear, and the probability of leaving descendants greatly increases when passing to the next generation.
The cart that performed second best in the previous generation has an increased chance of leaving offspring when passing to the next generation.
For achievements, you get points proportional to your progress on the track if you don't pass the track, and if you pass the track, you get a higher score the shorter the time you passed.

Currently implemented features include:

 Specter mode (default)
Follows the top performing karts from previous generations, allowing you to see your progress.
Press the a key in play mode to switch to specter mode.
In Specter mode, you can view up to 6 record update logs at once. You can also view more logs (up to 25) by pressing l.

 play mode
If you press the a key in Specter mode, it will switch to play mode, and you can race with the karts you are training yourself.
The controls are as follows:
Up Arrow: Excel
left arrow: turn left
right arrow: turn right
z key: drift
While using Drift, you won't accelerate even with the booster turned on, you can turn more sharply than usual, and the booster gauge will recharge.
With the booster on, the average speed increases by 100% in about 1.7 seconds.

 key viewer
You can see how the key of the cart you are currently following is entered in the lower left corner. Each button represents Drift, Left Turn, Accel, Right Turn.

 save/load
You can save the current progress by pressing ctrl + s. Saved save files can be selected to proceed when running a new program.
As an example file, I've saved a save file with a record of 19.72 seconds, which was the best performance.

 Thank you for your interest in my first post-college project. Since it is a project conducted in pure Python without using a game engine, it is better than a genetic algorithm.
It sounds like you put a lot more time into your learning environment and optimization. Because it is its own engine, it can have bugs and is not efficient, but personally it was a very fun project.
In the future, when I have time, I plan to make such a short project once in a while. I would like to ask for your anticipation and support for various projects such as artificial intelligence/games that I will proceed with. thank you

If you have any questions about the project, you can contact us at the contact details below.
developer Contact: kevin051211@kaist.ac.kr