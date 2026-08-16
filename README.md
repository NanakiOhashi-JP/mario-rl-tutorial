# ようこそ

PyTorchの公式チュートリアルには、強化学習で『Super Mario Bros.』を攻略するものがあります。ただ、初めて強化学習に触れる人にとっては、少し難しく感じる部分もあります。

そこで本チュートリアルでは、もう少し手前の段階から、実際にコードを動かしながら学んでいきます。基本的なPythonの読み書きができることを前提に、なるべく丁寧に進めます。わからない言葉や処理が出てきたら、ぜひ立ち止まって調べてみてください。それも大切な学習の一部です。

このチュートリアルは、次の順番で進めます。

1. gym-super-mario-brosを触る
    - ゲームを起動する
    - ボタンを押してマリオを動かす
    - 画面や座標など、環境から何を取得できるか見る
    - 報酬（reward）がどのように決まるか確認する
2. ルールベースで動かす
    - まずは右へ進む
    - 穴や敵を検知する方法を考える
    - 条件に応じてジャンプする
    - 1-1をどこまで攻略できるか試す
3. DQNに置き換える
    - ルールベースの処理を振り返る
    - 「この状態ならこの行動」をニューラルネットワークに学習させる
    - 状態（state）・行動（action）・報酬（reward）のつながりを理解する

# gym-super-mario-brosを触ろう

まずはプロジェクトを作りましょう。本チュートリアルでは、Pythonのプロジェクト管理に`uv`を使います。仮想環境やパッケージをまとめて管理でき、動作も速い便利なツールです。

:::note info
`uv`のインストール手順は[公式ドキュメント](https://docs.astral.sh/uv/getting-started/installation/)を参照してください。
:::

次のコマンドを順番に実行します。

```bash
uv init mario-rl-tutorial
cd mario-rl-tutorial
uv python pin 3.13
uv add gym-super-mario-bros gymnasium
```

> RLは、Reinforcement Learning（強化学習）の略です。

旧`gym`を使用するバージョンも存在しますが、本チュートリアルではGymnasium対応版を使います。このバージョンの`gym-super-mario-bros`にはPython 3.13以上が必要です。

機械学習を始める前に、まずは学習対象がどのようなものなのかを知る必要があります。材料を知らないまま料理はできませんからね。

それでは、`main.py`を作成して、実際の動作を見てみましょう。

:::note warn
VS Codeでは、`uv`の仮想環境が自動で選択されないことがあります。`Ctrl + Shift + P`でコマンドパレットを開き、`Python: インタープリターの選択`から、このプロジェクトの仮想環境を選んでください。
:::

## はじめてのgym

`main.py`に次のコードを書きます。

```python:main.py
import gymnasium as gym
import gym_super_mario_bros

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env.reset()

env.render()

input("Enterで終了")

env.close()
```

いきなり知らないものがたくさん出てきましたね。大丈夫です。上から順番に見ていきましょう。

`gymnasium`は、強化学習用の環境を共通の形式で扱うための土台となるライブラリです。もう少し噛み砕くと、**「ゲームやシミュレータをAIから操作するための共通インターフェース」**です。もともとは`gym`というライブラリが広く使われていましたが、現在はその後継である`gymnasium`がよく使われています。

一方、`gym_super_mario_bros`は、マリオのゲーム環境をGymnasiumに登録する役割を持っています。コード内で名前を直接使っていなくても、importは必要です。

```python
env = gym.make("SuperMarioBros-1-1-v0")
```

このコードは、「1-1のマリオの世界を、Pythonから操作できるオブジェクトとして作って」と指示しています。強化学習では、このような環境（environment）を略して`env`と書くのが定番です。

続いて、

```python
env.reset()
```

でゲームを最初の状態に戻します。そして、

```python
env.step(action)
```

と書くと、環境に「この操作を1回実行して」と指示できます。`action`には、右を押す、ジャンプするといった操作が入ります。これは次の節で実際に試します。

`gym.make()`に渡している`render_mode`は、環境をどのような形で描画するかを決めるオプションです。

- `"human"`：人が見られるウィンドウとして表示する
- `"rgb_array"`：画面を画像データとして受け取る

今回は実際のゲーム画面を見たいので、`"human"`を指定しています。また、このマリオ環境では、画面を表示するために`env.render()`を明示的に呼び出す必要があります。

最後の部分も見ておきましょう。

```python
input("Enterで終了")
env.close()
```

`input()`でEnterキーが押されるまでプログラムを待機させ、そのあと`env.close()`でゲーム画面を閉じています。これがないと、画面が開いてもすぐにプログラムが終了してしまいます。

これで、マリオの世界をPythonから起動できました。次は、いよいよマリオを動かしてみましょう。

## Let's-a go!

さて、いよいよマリオを動かします。まずは難しいことを考えず、とにかく右へ走らせてみましょう。

### マリオの操作をシンプルにする

ファミコンのコントローラーには、十字キーやAボタン、Bボタンなどがありますね。複数のボタンを同時に押す場合もあるため、その組み合わせは全部で256通り。いきなり全部から選べと言われても、AIでなくても困ります。

しかし、「上 + 左 + A + B」のように、マリオを操作する上でほとんど使わない組み合わせまで考えるのは大変です。そこで登場するのが、`SIMPLE_MOVEMENT`と`JoypadSpace`です。

#### SIMPLE_MOVEMENTとは

`SIMPLE_MOVEMENT`は、マリオでよく使うボタン操作を7つにまとめた「操作リスト」です。例えば、右へ進む、右へ進みながらジャンプする、といった実用的な操作だけが用意されています。「左を押しながら右を押す」といった、マリオも困惑しそうな操作は最初から候補に入れません。

つまり、`SIMPLE_MOVEMENT`は「AIが選べる操作のメニュー」です。

#### JoypadSpaceとは

`JoypadSpace`は、ゲーム環境を包み、AIから受け取った行動番号を実際のボタン入力に変換するラッパーです。ラッパー（wrapper）とは、元の環境そのものを書き換えず、外側から機能を追加したり、入出力を扱いやすく変換したりする仕組みです。

今回は、`JoypadSpace`が次のような翻訳役になります。

```text
env.step(1) → SIMPLE_MOVEMENTの1番 → 右ボタンを押す
```

`SIMPLE_MOVEMENT`が操作の選択肢を用意し、`JoypadSpace`が選ばれた番号をゲームに伝える。この2つを組み合わせることで、256通りあった行動を7つに絞れます。

まず、次のimportを追加します。

```python
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
```

そして、`gym.make()`で作った環境を`JoypadSpace`で包みます。

```python
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```

`SIMPLE_MOVEMENT`で使える行動は次のとおりです。

| 番号 | 操作 |
| ---: | --- |
| `0` | 何も押さない |
| `1` | 右 |
| `2` | 右 + ジャンプ |
| `3` | 右 + ダッシュ |
| `4` | 右 + ジャンプ + ダッシュ |
| `5` | ジャンプ |
| `6` | 左 |

今回は右へ進みたいので、行動番号の`1`を使います。

### ゲームを1フレームずつ進める

環境をリセットする部分も、少し書き換えましょう。

```python
observation, info = env.reset()
```

`env.reset()`は、ゲームを最初の状態に戻すだけでなく、次の2つの値を返します。

- `observation`：リセット直後のゲーム画面
- `info`：マリオの座標など、環境に関する追加情報

今はまだこれらを使いませんが、後で必要になるので受け取っておきます。

次に、ループを使ってゲームを1,000フレーム進めます。

```python
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(1)
    env.render()
```

`env.step(1)`は、環境に「右ボタンを押して1フレーム進めて」と指示しています。呼び出すたびに、次の5つの値が返ってきます。

- `observation`：行動後のゲーム画面
- `reward`：その行動で得られた報酬
- `terminated`：ミスやゴールなどで、エピソードが終了したか
- `truncated`：制限時間など、環境側の条件で打ち切られたか
- `info`：マリオの座標やスコアなどの追加情報

強化学習では、基本的にこの「状態を見て行動し、報酬を受け取る」という流れを繰り返します。まだAIは登場していませんが、すでに強化学習の基本の形になっています。

### 表示速度を整える

このままではループが可能な限り速く実行されるため、ゲームが一瞬で進んでしまいます。約1秒に60フレームの速度で表示するため、`time.sleep()`を追加します。

```python
import time

# 中略

time.sleep(1 / 60)
```

`1 / 60`秒ずつ待機することで、人の目でも追いやすい速度になります。

### ゲームが終わったらリセットする

マリオがミスをしたり、環境が制限時間に達したりした後は、そのまま`step()`を続けることができません。どちらかの終了フラグが`True`になったら、環境をリセットします。

```python
if terminated or truncated:
    observation, info = env.reset()
```

これで、マリオが右へ進み、ミスをしたら最初からやり直すプログラムになりました。

### 完成したコード

ここまでの内容をまとめると、`main.py`は次のようになります。

```python:main.py
import time

import gymnasium as gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(1)
    env.render()
    time.sleep(1 / 60)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

実行してみましょう。

```bash
uv run python main.py
```

マリオが自動で右へ進んだら成功です。もちろん、今のマリオは右に走り続けることしか知らないので、穴やクリボーにぶつかってもお構いなしです。次は、環境から取れる情報を観察して、どうすればうまく避けられるか考えていきましょう。

## Mamma mia!

右へ進むことしか知らないマリオは、当然のように最初のクリボーにぶつかります。いくら主人公でも、自動で敵を避けてはくれません。

ここで、最初のクリボーに「ジャンプのタイミングを教えてくれる先生」になってもらいましょう。マリオが動き始めてから、何フレーム後にぶつかるのかを計ります。

### 生存時間を計る

生存時間を数えるため、`survival_time`という名前の変数を用意します。

```python
survival_time = 0
```

1フレーム進むたびに、値を1増やします。

```python
observation, reward, terminated, truncated, info = env.step(action)
env.render()
time.sleep(1 / 60)
survival_time += 1
```

変数名は`survival_time`ですが、ここで数えているのは秒数ではなく、**生存したフレーム数**です。名前は少し大げさですが、仕事はまじめなカウンターです。1秒間に約60フレーム進めているので、例えば160フレームは、およそ`160 ÷ 60 = 2.67`秒に相当します。

マリオがミスをしたら、その時点の`survival_time`を表示します。

```python
if terminated or truncated:
    print(terminated, truncated)
    print(f"Survival Time: {survival_time}")
    input("Press Enter to continue...")
    observation, info = env.reset()
    survival_time = 0
```

`print(terminated, truncated)`では、2つの終了フラグのどちらが`True`になったかも確認しています。`input()`を入れているのは、ミスをした直後で一度プログラムを止め、ターミナルの計測結果を確認しやすくするためです。Enterキーを押すとゲームをリセットし、`survival_time`も`0`に戻して次の計測を始めます。

実行してみると、今回の条件では`survival_time`がおよそ`160`のときに、最初のクリボーと接触することがわかりました。我々の最初の宿敵です。

### 計測結果を行動に反映する

クリボーと接触するのが160フレーム付近なら、その少し前にジャンプすれば避けられそうです。そこで、`survival_time`が`150`のときだけ、行動を変えてみます。

```python
if survival_time == 150:
    action = 2
else:
    action = 1
```

`SIMPLE_MOVEMENT`では、行動`1`が「右」、行動`2`が「右 + ジャンプ」でした。したがって、この条件分岐は次のような指示になります。

- `survival_time == 150`のとき：右へ進みながらジャンプする
- それ以外：右へ進む

先ほどの`env.step(1)`も、変数`action`を使う形に書き換えます。

```python
observation, reward, terminated, truncated, info = env.step(action)
```

これが、**状況に応じて行動を選ぶ**ための最初の一歩です。まだ画面や敵の位置を見て判断しているわけではなく、マリオの頭の中は「150になったら跳べ」というメモ1枚です。これをルールベースのプログラムと呼びます。

それでも、「計測する→ルールを作る→行動を変える」という、強化学習につながる考え方を体験できます。

実行すると、マリオがクリボーの少し手前でジャンプし、無事に飛び越えます。さよなら、クリボー。Yahoo!

### ここまでのコード

```python:main.py
import time

import gymnasium as gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

survival_time = 0

for _ in range(1000):
    if survival_time == 150:
        action = 2
    else:
        action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)
    survival_time += 1

    if terminated or truncated:
        print(terminated, truncated)
        print(f"Survival Time: {survival_time}")
        input("Press Enter to continue...")
        observation, info = env.reset()
        survival_time = 0

env.close()
```

## 現実問題
じゃあすべてのマリオの動きを0から100まで設定して…ってのは無理がありますね．
条件分岐も大変だし，何よりクリボーが不規則に動いたら，違うステージになったら，とてもじゃないけど人手でやるには厳しいですね．
では，どうすればどんなステージでも，どんなクリボーが来ようともクリアを目指すマリオを作れるでしょう？
### 「見ろ！マリオ！」
我々がプレイするときどのように判断するでしょう？
そうですね，「見て」判断します．
マリオにも見て判断してもらいましょう．我々には画像データが与えられてましたね！
