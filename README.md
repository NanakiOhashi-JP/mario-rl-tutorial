# マリオで学ぶ強化学習

## It's-a me, Mario!

PyTorchの公式チュートリアルには、強化学習で『Super Mario Bros.』を攻略するものがあります。ただ、初めて強化学習に触れる人にとっては、少し難しく感じる部分もあります。

そこで本チュートリアルでは、もう少し手前の段階から、実際にコードを動かしながら学んでいきます。基本的なPythonの読み書きができることを前提に、なるべく丁寧に進めます。わからない言葉や処理が出てきたら、ぜひ立ち止まって調べてみてください。それも大切な学習の一部です。

このチュートリアルは、次の順番で進めます。

1. 第1章：ルールベースでマリオを動かす
    - `gym-super-mario-bros`でゲームを起動する
    - ボタンを押してマリオを動かす
    - 画面や座標など、環境から取得できる情報を見る
    - まずは右へ進む
    - タイミングや座標でジャンプする
    - 画像から敵を検知する
2. 第2章：ルールをDQNに置き換える
    - ルールベースの処理を振り返る
    - 「この状態ならこの行動」をニューラルネットワークに学習させる
    - 状態（state）・行動（action）・報酬（reward）のつながりを理解する

# 第1章 ルールベースでマリオを動かそう

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

## 1. ゲームを起動しよう

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

一方、`gym_super_mario_bros`は、マリオのゲーム環境をGymnasiumに登録する役割を持っています。コード内で名前を直接使っていなくても、`import`は必要です。

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

## 2. マリオを動かそう — Let's-a go!

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

`1 / 60`秒ずつ待機することで、人の目でも追いやすい速度になります。実際には描画などの処理時間も加わるため、厳密な60 FPSではなく、おおよその目安と考えてください。

### ゲームが終わったらリセットする

マリオがミスをしたり、環境が制限時間に達したりした後は、そのまま`step()`を続けることができません。どちらかの終了フラグが`True`になったら、環境をリセットします。

```python
if terminated or truncated:
    observation, info = env.reset()
```

これで、マリオが右へ進み、ミスをしたら最初からやり直すプログラムになりました。

### 右へ進むコードの全体像

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

## 3. 最初のクリボーを避けよう — Mamma mia!

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

### タイミングでジャンプするコードの全体像

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

## 4. `info`を使って状況を知ろう

では、1-1をクリアするまでの動きを、最初から最後まですべて手作業で決めて……というのは、さすがに無理があります。条件分岐は増え続けますし、クリボーが少し違う動きをしたり、ステージが変わったりすれば、すぐに使えなくなってしまいます。これではマリオの攻略より、`if`文の攻略のほうが大変です。

そういえば、`env.reset()`や`env.step()`は`info`を返していました。`info`には、マリオの座標、残り時間、スコア、画面内にいる敵の種類など、環境に関する補助情報が入っています。ゲーム画面そのものではありませんが、上手に使えばマリオをもう少し賢くできそうです。

### `info`の中身を見る

まずは、リセット直後の`info`を表示してみましょう。

```python
from pprint import pprint

observation, info = env.reset()

selected_info = {
    "x_pos": info["x_pos"],
    "y_pos": info["y_pos"],
    "enemy_types": info["enemy_types"],
    "time": info["time"],
    "score": info["score"],
    "death": info["death"],
}

pprint(selected_info, sort_dicts=False)
```

`pprint()`は、辞書などのデータを読みやすく整形して表示する関数です。実行すると、次のように表示されます。

```text
{
    'x_pos': 40,
    'y_pos': 79,
    'enemy_types': (0, 0, 0, 0, 0),
    'time': 400,
    'score': 0,
    'death': False,
}
```

ここで重要なのは、`info`に敵の正確な位置までは入っていないことです。`enemy_types`を見れば画面内にいる敵の種類はわかりますが、その敵がマリオのすぐ目の前にいるのか、まだ画面の端にいるのかまではわかりません。惜しい。

### 時間ではなく位置で判断する

前の節では、生存フレーム数をもとにジャンプしました。ただし、それまでに別の動作をしたり、何かに引っかかったりすると、同じフレームでもマリオの位置は変わってしまいます。時間ベースは、寄り道には弱いのです。

そこで今回は、経過フレームではなく、マリオの位置をもとにジャンプのタイミングを決めてみましょう。

```python
if 270 <= info["x_pos"] <= 280:
    action = 2
else:
    action = 1
```

`x_pos`は、マリオの横方向の座標です。この値が270から280の間に入ったら、右へ進みながらジャンプします。時間ではなく場所を基準にしたので、途中で少し遅れても、同じ場所でジャンプできます。マリオ、少し賢くなりました。

### 動けなくなったらジャンプする

さらに、マリオの横座標が動かない状態が続いたら、目の前に土管などの障害物があると仮定してジャンプさせてみましょう。

```python
former_x_pos = None
stuck_frames = 0

for _ in range(1000):
    current_x_pos = info["x_pos"]

    if former_x_pos == current_x_pos:
        stuck_frames += 1
    else:
        stuck_frames = 0

    if 270 <= current_x_pos <= 280 or stuck_frames >= 10:
        action = 2
    else:
        action = 1

    former_x_pos = current_x_pos
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    if terminated or truncated:
        # 中略
        former_x_pos = None
        stuck_frames = 0
```

`former_x_pos`に前回の横座標を保存し、現在の座標と比較しています。ただし、`x_pos`は整数なので、普通に移動していても前のフレームと同じ値になることがあります。1回一致しただけで「行き止まった！」と判定するのは、少しせっかちです。

そこで、座標が変わらないフレーム数を`stuck_frames`で数え、10フレーム連続で動かなかったときだけジャンプします。これで、スタート直後や通常の移動中に、勘違いしてジャンプする回数を減らせます。土管の前で行き止まっても、しばらく考えてから飛び越える。慎重派のマリオです。

さて、土管は越えられましたが、次のクリボーはどうでしょう。また座標を調べて、その手前でジャンプさせますか？

できなくはありませんが、この先もたくさんのステージで遊ぶことを考えると、すべての座標を手作業で登録するのは現実的ではありません。それでは「賢い」というより、攻略本を一字一句暗記したマリオです。

座標を暗記するのではなく、目の前の状況を見て判断できるようにする必要がありそうです。

## 5. 画像からクリボーを見つけよう

座標の丸暗記では、別のステージに応用できません。そこで次は、マリオにきちんと前を見てもらいます。

ゲーム画面の情報は、`env.reset()`や`env.step()`が返す`observation`に入っていましたね。まずは、その中身を見てみましょう。

### `observation`の中身を見る

画面をウィンドウではなく画像データとして扱うため、`render_mode`を`"rgb_array"`に変更します。

```python
env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="rgb_array"
)

observation, info = env.reset()

print(observation)
```

実行すると、次のような値が表示されます。

```text
[[[104 136 252]
  [104 136 252]
  [104 136 252]
  ...
  [104 136 252]
  [104 136 252]
  [104 136 252]]
...
```

なんだか暗号みたいですが、これはゲーム画面を構成する全ピクセルの色情報です。`observation`の形も確認してみましょう。

```python
print(observation.shape)
```

```text
(240, 256, 3)
```

最初の`240`は縦のピクセル数、次の`256`は横のピクセル数、最後の`3`はRGBの3色を表しています。つまり、`observation`の形は`(高さ, 横幅, 色)`です。

1ピクセルには、赤（R）、緑（G）、青（B）の3つの値があり、それぞれ`0`から`255`までの値を取ります。例えば`[104, 136, 252]`は、赤が104、緑が136、青が252の色です。見慣れない数字ですが、これがあの青空の正体です。

### OpenCVでクリボーを探す

この画像データからクリボーを探すため、OpenCVという画像処理ライブラリを使います。まずはプロジェクトに追加しましょう。

```bash
uv add opencv-python
```

OpenCVにはさまざまな画像処理機能がありますが、今回使うのは「テンプレートマッチング」です。お手本となる小さな画像を用意し、大きな画像の中からそれと似た部分を探します。クリボーの顔写真を持って、画面の中を聞き込みして回るようなものです。

テンプレートには、あらかじめ`observation`から原寸大で切り出したクリボーの画像を使います。画像の切り出し自体は強化学習の本題ではないため、本チュートリアルでは用意済みの素材を配布します。次の場所へ保存してください。

```text
templates/
└── goomba.png
```

お手本の切り抜きで苦戦するより、マリオに苦戦してもらいましょう。

次のコードで、クリボーの位置を探します。

```python
import cv2

# 中略

template = cv2.imread("templates/goomba.png")

if template is None:
    raise FileNotFoundError("クリボーの画像が見つかりません")

# 中略

for _ in range(1000):
    action = 1
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    _, score, _, position = cv2.minMaxLoc(result)

    if score >= 0.8:
        print(f"クリボー発見！: x = {position[0]}, y = {position[1]}")

    # 以下、中略
```

順番に見ていきましょう。

- `cv2.imread()`：画像ファイルを読み込む。`imread`は「image read」の略です。
- `cv2.cvtColor()`：色の並び順をRGBからBGRへ変換する。`observation`はRGB順ですが、OpenCVは画像をBGR順で扱うためです。
- `cv2.matchTemplate()`：ゲーム画面の各位置とテンプレートを比較し、似ている度合いを表す2次元のスコア表を返す。
- `cv2.minMaxLoc()`：スコア表から最小値、最大値、それぞれの位置を取得する。

`cv2.minMaxLoc()`の返り値は、`(最小値, 最大値, 最小値の位置, 最大値の位置)`です。今回は「最もクリボーに似ている場所」だけが欲しいので、最大値とその位置を`score`と`position`で受け取っています。使わない値は`_`で受け取り、「ここは今回見ません」という意思表示をしています。

`cv2.TM_CCOEFF_NORMED`を使った場合、`score`が`1.0`に近いほどテンプレートとよく似ています。ここでは`0.8`以上をクリボーと判定しています。`0.8`は確率80%という意味ではなく、あくまでも「十分に似ている」と判定するためのしきい値です。

これで、クリボーの画面上の位置を取得できました。

### クリボーが近づいたらジャンプする

クリボーの位置がわかったので、次はマリオとの距離を調べます。

ここで注意したいのが、座標系の違いです。`info["x_pos"]`はステージ全体におけるマリオの横座標ですが、テンプレートマッチングが返す`position`は現在の画面上の座標です。ステージ全体の地図と、テレビ画面の上の位置をそのまま比べることはできません。

幸い、`info["left_x_pos"]`には、マリオが画面の左端から何ピクセルの位置にいるかが入っています。これなら、OpenCVが見つけたクリボーの横座標と直接比較できます。

```python
mario_x = info["left_x_pos"]
distance = goomba_x - mario_x

if 20 < distance < 50:
    action = 2
else:
    action = 1
```

クリボーがマリオの20から50ピクセル前方にいるとき、右へ進みながらジャンプします。「クリボーを見つけた」だけではなく、「クリボーが近づいた」と判断できるようになりました。

ここまでのコードをまとめると、次のようになります。

```python:main.py
import time

import cv2
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

template = cv2.imread("templates/goomba.png")

if template is None:
    raise FileNotFoundError("クリボーの画像が見つかりません")

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human"
)

env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

goomba_x = None

for _ in range(1000):
    mario_x = info["left_x_pos"]

    if goomba_x is not None and 20 < goomba_x - mario_x < 50:
        action = 2
    else:
        action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1 / 60)

    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    _, score, _, position = cv2.minMaxLoc(result)

    if score >= 0.8:
        goomba_x = position[0]
        print(f"クリボー発見！: x = {goomba_x}, score = {score:.3f}")
    else:
        goomba_x = None

    if terminated or truncated:
        input("Press Enter to continue...")
        observation, info = env.reset()
        goomba_x = None

env.close()
```

実行すると、マリオは画面内のクリボーを見つけ、近づいたところでジャンプします。ついに、攻略本の座標ではなく、自分の目でクリボーを見て避けられるようになりました。

もっとも、今のマリオが覚えているのはクリボーの顔だけです。土管も穴も、別の敵もまだ知りません。お手本画像と条件分岐を増やせば対応できますが、このままではまた`if`文の山ができそうです。

とはいえ、ここまでの集大成をまとめてみましょう。

### これまでのルールを組み合わせる

```python:main.py
import time
from pathlib import Path

import cv2
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "goomba.png"
MATCH_THRESHOLD = 0.8
MIN_GOOMBA_DISTANCE = 20
MAX_GOOMBA_DISTANCE = 50
STUCK_FRAME_THRESHOLD = 10

template = cv2.imread(str(TEMPLATE_PATH))
if template is None:
    raise FileNotFoundError(f"クリボーの画像が見つかりません: {TEMPLATE_PATH}")

env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="human",
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()

goomba_x = None
former_x_pos = None
stuck_frame_count = 0

try:
    for _ in range(1000):
        current_x_pos = info["x_pos"]

        # 同じステージ座標から動けないフレーム数を数える。
        if former_x_pos is not None and former_x_pos == current_x_pos:
            stuck_frame_count += 1
        else:
            stuck_frame_count = 0

        mario_x = info["left_x_pos"]
        goomba_distance = None if goomba_x is None else goomba_x - mario_x

        is_goomba_nearby = (
            goomba_distance is not None
            and MIN_GOOMBA_DISTANCE < goomba_distance < MAX_GOOMBA_DISTANCE
        )
        is_stuck = stuck_frame_count >= STUCK_FRAME_THRESHOLD

        # クリボーが近い、または10フレーム連続で動けないときにジャンプする。
        action = 2 if is_goomba_nearby or is_stuck else 1

        former_x_pos = current_x_pos
        observation, _, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1 / 60)

        # テンプレートマッチングでクリボーの画面座標を取得する。
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(result)

        if score >= MATCH_THRESHOLD:
            goomba_x = position[0]
        else:
            # 前フレームの検出位置を使い続けない。
            goomba_x = None

        if terminated or truncated:
            input("Press Enter to continue...")
            observation, info = env.reset()
            goomba_x = None
            former_x_pos = None
            stuck_frame_count = 0
except KeyboardInterrupt:
    pass
finally:
    env.close()
```

少し今までより丁寧に書いていますが、やっていることは「マリオが動かなくなったらジャンプ」と「クリボーが近づいたらジャンプ」の2つだけです。

## 第1章のまとめ

この章では、マリオを起動するところから始め、次の順番で行動の決め方を改良してきました。

1. 常に右へ進む
2. 決めたタイミングでジャンプする
3. `info`からマリオの位置を取得し、座標で行動を変える
4. 動けない状態を検知し、土管をジャンプで越える
5. `observation`の画像からクリボーを見つけ、近づいたらジャンプする

最初は「右へ進め」しか知らなかったマリオが、ずいぶん自分で状況を見られるようになりました。とはいえ、ルールやテンプレートを用意したのは、すべて人間です。

敵やステージが増えるたびに条件分岐を書き足すのではなく、どの行動がよかったのかをマリオ自身に学んでもらえないでしょうか。次の章から、いよいよ強化学習に足を踏み入れます。

# 第2章 強化学習への扉

## 1. ご褒美だよ、マリオ

さて、これまでのマリオは何を目的に走っていたのでしょうか。ピーチ姫を取り返すため？ それとも、ただのマラソン？

第1章でやっていたのは、人間が決めた条件に従ってマリオを動かすことでした。クリボーは避けられるようになりましたが、マリオ自身は「なぜジャンプするのか」を知りません。言われたから跳んでいるだけです。いくら主人公でも、これでは少し寂しい。

ここからは、マリオに明確な目的を持ってもらいます。制限時間の中でできるだけ遠くへ進み、最終的にゴールすることです。

でも、その目的をどうやって教えればよいのでしょうか。

そこで使うのが、**報酬（reward）** です。マリオの行動に対して、よい結果にはプラスの報酬、悪い結果にはマイナスの報酬を与えます。

- 右へ進んだ：よいこと
- ゴールした：とてもよいこと
- クリボーにぶつかった：悪いこと
- 穴に落ちた：とても悪いこと

アイテムを取るのはよいことに見えますが、最短時間でのゴールが目的なら、寄り道になるかもしれません。何を褒めるかによって、マリオの行動は変わります。教育方針は大切です。

強化学習では、この報酬を長い目で見てできるだけ大きくするように行動を学びます。ただし、いきなりマリオの画面で考えると大変なので、まずはずっと小さなゲームで仕組みを見てみましょう。

## 2. Q学習の考え方

:::note info
ここから少しだけ用語が増えますが、暗記する必要はありません。「行動に点数をつけて、よかった行動を覚えるんだな」とつかめれば十分です。
:::

### 4マスの世界で考える

次のような、とてもシンプルなゲームを考えます。

```text
[S] [1] [2] [G]
```

- `[S]`：スタート地点
- `[1]`、`[2]`：途中のマス
- `[G]`：ゴール地点

選べる行動は「左へ進む」と「右へ進む」の2つだけ。ルールも「ゴールに到達したら報酬`+1`」だけです。土管もクリボーもいません。ほぼマリオですね。たぶん。

この「今どのマスにいるか」のように、行動を決めるために使う現在の状況を**状態（state）**と呼びます。このゲームでは、`[S]`、`[1]`、`[2]`のどこにいるかが状態です。

最初は、プレイヤーがすべて右へ進んだとします。

1. `[S]`から`[1]`へ進む：報酬`0`
2. `[1]`から`[2]`へ進む：報酬`0`
3. `[2]`から`[G]`へ進む：報酬`+1`

ゴールしたので、めでたしめでたし。と言いたいところですが、学習の視点ではここからが本番です。

### Qテーブルに行動の点数を記録する

プレイヤーは、「この場所では、どちらへ進むのがよさそうか」を点数で覚えます。この行動の点数を**Q値**、点数を書き込む表を**Qテーブル**と呼びます。

```text
Q（状態, 行動）
= その状況で、その行動を選ぶとどれくらいよさそうか
```

学習を始める前は何も知らないので、すべてのQ値を`0`にしておきます。

| 状態 | 左 | 右 |
| --- | ---: | ---: |
| `[S]` | 0 | 0 |
| `[1]` | 0 | 0 |
| `[2]` | 0 | 0 |

先ほどのプレイでは、`[2]`から右へ進んだときに、報酬`+1`を受け取りました。そこで、この行動のQ値を`1.0`に更新します。

| 状態 | 左 | 右 |
| --- | ---: | ---: |
| `[S]` | 0 | 0 |
| `[1]` | 0 | 0 |
| `[2]` | 0 | 1.0 |

これで、「`[2]`にいるときは、右へ進むとよさそうだ」と覚えました。

しかし、`[S]`や`[1]`での行動はまだ`0`点のままです。ゴール直前だけ賢くても、そこまでたどり着けなければ意味がありません。

### ゴールの価値を手前へ伝える

もう一度、同じゲームをプレイしてみましょう。

`[1]`から右へ進んだとき、すぐにもらえる報酬は`0`です。しかし、移動先の`[2]`では「右へ進めば`1.0`点」ということを、プレイヤーはすでに知っています。

```text
[1]から右へ進む
    ↓
今回の報酬は0
    ↓
でも、移動先の[2]には1.0点の行動がある
```

つまり、`[1]`から右へ進む行動も、将来的にはよい行動です。

ただし、遠くにあるゴールと、すぐ隣にあるゴールを同じ点数にはしません。同じ報酬`+1`をもらえるなら、なるべく短い道を選んでほしいからです。

そこで、ゴールの点数を1マス手前へ伝えるたびに、`0.9`倍して少しだけ小さくします。この`0.9`を**割引率**と呼びます。ゴールが遠いほど途中で失敗する可能性も増えるので、遠くの報酬は少し控えめに見積もるわけです。

```text
Q（[1], 右）
= 今回の報酬 + 0.9 × 移動先で一番高いQ値
= 0 + 0.9 × 1.0
= 0.9
```

Qテーブルは次のようになります。

| 状態 | 左 | 右 |
| --- | ---: | ---: |
| `[S]` | 0 | 0 |
| `[1]` | 0 | 0.9 |
| `[2]` | 0 | 1.0 |

さらにもう一度プレイすると、今度は`[S]`から右へ進む行動にも、将来の点数が伝わります。

```text
Q（[S], 右）
= 0 + 0.9 × 0.9
= 0.81
```

| 状態 | 左 | 右 |
| --- | ---: | ---: |
| `[S]` | 0 | 0.81 |
| `[1]` | 0 | 0.9 |
| `[2]` | 0 | 1.0 |

ゴールに近いほど、右へ進む行動のQ値が高くなりました。ゲームを繰り返すたびに、ゴールでもらった報酬が少しずつ手前へ伝わっています。報酬のバケツリレーです。

:::note info
ここでは流れを見やすくするため、計算した点数をそのままQテーブルへ書き込んでいます。実際には、あとで登場する「学習率」を使って少しずつ書き換えます。
:::

### ときどきは知らない行動も試す

Qテーブルができたら、基本的には現在の状態でQ値が最も高い行動を選べばよさそうです。

しかし、学習を始めた直後は、ほとんどのQ値が`0`です。その状態で「右」しか試さなければ、もしかすると「左」の先にもっとよい結果があっても気づけません。

そこで、ときどきはQ値を無視して、ランダムな行動を選びます。

- ほとんどの場合：Q値が最も高い行動を選ぶ
- ときどき：ランダムな行動を選ぶ

これを**ε-greedy（イプシロン・グリーディ）法**と呼びます。`greedy`は「今のところ一番よさそうな行動を選ぶ」、`ε`は「たまにランダムな行動を試す確率」です。例えば`ε = 0.1`なら、約90%はQ値が一番高い行動を選び、約10%はランダムに動きます。

知らない行動を試すことを**探索**、覚えたよい行動を使うことを**活用**と呼びます。難しそうな名前ですが、作戦は「基本はいつもの、たまに冒険」です。

マリオにも、たまには寄り道が必要です。

## 3. これがQ学習

ここまでに行ったことを振り返ってみましょう。

1. 現在の状態を確認する
2. 行動を選ぶ
3. 報酬と次の状態を受け取る
4. 次の状態で得られそうなQ値も考える
5. Qテーブルを更新する

実は、これが**Q学習**です。名前だけ聞くと難しそうですが、やっているのは「試して、結果を見て、行動の点数を書き換える」の繰り返しです。

ここまで表を使って説明した点数の更新を、数式でまとめると次のようになります。

$$
Q(s,a) \leftarrow Q(s,a)
+ \alpha \left[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\right]
$$

急に強そうな式が出てきましたが、新しい話ではありません。今は暗記せず、登場人物だけ見ておきましょう。

| 記号 | 意味 |
| --- | --- |
| $s$ | 現在の状態 |
| $a$ | 現在の状態で選んだ行動 |
| $r$ | 行動の結果、受け取った報酬 |
| $s'$ | 行動後の次の状態 |
| $\max_{a'}Q(s',a')$ | 次の状態にある、一番高い行動の点数 |
| $\alpha$ | 学習率。今回の結果で、点数をどれくらい書き換えるか |
| $\gamma$ | 割引率。先ほど使った`0.9`で、未来の点数をどれくらい残すか |

数式全体を日本語に戻すと、次のようになります。

```text
新しいQ値
= 今までのQ値
  + 学習率 ×（今回と未来を合わせた点数 - 今までの予想）
```

つまり、「今回の結果は予想よりよかった？ 悪かった？ では点数を少し直そう」という式です。数式はいかついですが、やっていることはQテーブルの書き直しです。

## 4. Qテーブルをマリオに持たせられるか

4マスの世界では、「今いるマス」を状態としてQテーブルに書けました。では、マリオの状態をゲーム画面にすると、どうなるでしょうか。

マリオが受け取る1画面の形は、次のとおりでした。

```text
(240, 256, 3)
```

1画面だけで、`240 × 256 × 3 = 184,320`個もの数値があります。しかも、クリボーが1ピクセル動いたり、残り時間の数字が変わったりするだけで、別の画像になります。

Qテーブルへ保存するなら、その画像一つひとつに行を作り、第1章で用意した7種類の行動それぞれに点数を書かなければなりません。昨日見たクリボーと、今日1ピクセル動いたクリボーを、まったくの別人として覚えるようなものです。これでは表がいくらあっても足りません。

そもそも、空の色など、行動を決めるのにあまり必要ないピクセルもあります。このあたりは後で画像を小さくしたり、色を減らしたりして整理します。それでも、似た画面を別々に暗記するQテーブルだけでマリオを攻略するのは大変です。

## 5. Qテーブルをニューラルネットワークに置き換える

Qテーブルの弱点は、覚えた状態とまったく同じものしか探せないことでした。欲しいのは、「この画面、前にクリボーを避けたときと似ているな」と考える力です。

そこで、Qテーブルの代わりに**ニューラルネットワーク**を使います。マリオを何度もプレイさせながら、ゲーム画面、選んだ行動、その結果をセットで見せます。

例えば、「クリボーが目の前にいる画面でジャンプしたら先へ進めた」という経験を繰り返すと、ニューラルネットワークは似た画面でジャンプの点数を高くするようになります。

こうして学習したニューラルネットワークに現在の画面を見せると、「右は1.4点くらい、右 + ジャンプは5.8点くらいになりそうだ」と、各行動のQ値を**予測**します。Qテーブルから答えを探すのではなく、経験をもとに「たぶんこれくらい」と考えてもらうわけです。

```text
ゲーム画面
    ↓
ニューラルネットワーク
    ↓
各行動のQ値
```

例えば、7種類の行動に対して次のようなQ値を出力します。

| 行動 | 予測したQ値 |
| --- | ---: |
| 何もしない | 0.2 |
| 右 | 1.4 |
| 右 + ジャンプ | 5.8 |
| 右 + ダッシュ | 2.1 |
| 右 + ジャンプ + ダッシュ | 4.7 |
| ジャンプ | 0.8 |
| 左 | -1.2 |

この場合、最もQ値が高い「右 + ジャンプ」を選びます。

Q値を予測するニューラルネットワークを**Q-Network**と呼びます。そのネットワークを何層も重ねて作ったものが、**DQN（Deep Q-Network）** です。

> 大きすぎて作れないQテーブルの代わりに、ニューラルネットワークにQ値を予測してもらう

これがDQNの出発点です。Q学習そのものを捨てるのではなく、点数の覚え方を「表」から「予測」に変えています。これなら、初めて見る画面でも、過去の似た経験を使って行動を選べそうです。

ここまでで、ルールベースのマリオからDQNへ進む準備が整いました。次は、画面をニューラルネットワークに入れる前に、画像の大きさや色、時間方向の情報を整えていきます。

## 第2章のまとめ

この章では、マリオ自身に行動を選んでもらうための考え方を見てきました。

1. 行動の結果を報酬として受け取る
2. 状態と行動の組み合わせに、Q値という点数をつける
3. 経験をもとにQ値を書き換える
4. 基本はQ値の高い行動を選び、ときどき別の行動も試す
5. 大きすぎるQテーブルの代わりに、ニューラルネットワークでQ値を予測する

最後の仕組みがDQNでした。ただし、ゲーム画面をそのままニューラルネットワークへ渡すには、少し大きくて情報量も多すぎます。次の章では、マリオの画面をAIが扱いやすい形に整えます。

# 第3章 マリオの画面をAI向けに整えよう

いよいよDQNを作りたいところですが、その前に入力となるゲーム画面を整えます。大きな野菜を丸ごと鍋へ放り込むより、食べやすい大きさに切ったほうが料理しやすい。それと同じです。

元のゲーム画面には、マリオのひげや服の細かな模様まで入っています。しかし、次の行動を決めるために本当に知りたいのは、マリオや敵、足場がどこにあるかです。細かすぎる情報を減らし、学習しやすい形へ整える処理を**前処理**と呼びます。

この章では、次の順番でゲームの進め方と画面を整えます。

```text
同じ行動を4フレーム続ける
        ↓ 4フレーム分の報酬を合計する
(240, 256, 3) のカラー画像
        ↓ 小さくする
(84, 84, 3)
        ↓ グレースケールにする
(84, 84)
        ↓ 直近4枚を重ねる
(4, 84, 84)
        ↓ Tensorへ変換し、値を0〜1にそろえる
PyTorchで扱える状態
```

まず、PyTorchをプロジェクトへ追加しておきましょう。

```bash
uv add torch
```

## 1. 同じ行動を4フレーム続ける

これまでのコードでは、`env.step(action)`を1回呼ぶたびにゲームを1フレーム進めていました。しかし、隣り合った2枚の画面はほとんど同じです。わずかな変化しかない画面を毎回見せても、計算が増えるわりに新しい情報はあまり増えません。

さらに、行動を毎フレーム切り替えると、どの行動によって報酬を得たのか分かりにくくなります。例えば「右」で走り始めた直後に「A」へ切り替えると、右移動の勢いで進んだ報酬が「A」を選んだフレームに入ることがあります。手柄の持ち主が迷子です。

そこで、1回選んだ行動を4フレーム続けます。この処理を**Frame Skip（フレームスキップ）**と呼びます。

```python
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info
```

`SkipFrame`も、第3章で使っていくほかの前処理と同じラッパーです。`step()`の中で同じ行動を4回実行し、それぞれのフレームで受け取った報酬を`total_reward`へ足しています。

途中でゲームが終了した場合は、残りのフレームを進めず`break`します。やられたあとまでボタンを押し続けても、マリオとしては困るだけです。

```python
env = SkipFrame(env, skip=4)
```

`skip=4`なので、これ以降に`env.step(action)`を1回呼ぶと、内部では同じ行動で最大4フレーム進みます。返ってくる`reward`も、その間に得た報酬の合計です。

## 2. 画面を小さくする

元の画面は`240 × 256`ピクセルです。細かな絵までよく見えますが、画像が大きいほどニューラルネットワークが計算する量も増えます。

小さくしすぎれば敵や足場がつぶれてしまい、大きすぎれば学習に時間がかかります。今回は、情報の見やすさと扱いやすさのバランスを取り、`84 × 84`へ縮小します。

```python
from gymnasium.wrappers import ResizeObservation

env = ResizeObservation(env, (84, 84))
```

`84 × 84`は絶対の正解ではありません。DQNでよく使われる大きさで、マリオや敵を見分けられる程度の情報を残しながら、計算量を大きく減らせます。また、CNNは長方形の画像も扱えるため、「正方形でなければならない」というわけでもありません。今回は実装をシンプルにするため、この大きさを使います。

## 3. 色の情報を減らす

次は、カラー画像をグレースケールにします。色がなくても、マリオ、敵、土管、足場の位置は見分けられそうです。

カラー画像は、赤・緑・青の3つの値を持っています。グレースケールにすると明るさを表す1つの値だけになるため、ニューラルネットワークへ渡す情報をおよそ3分の1に減らせます。

```python
from gymnasium.wrappers import GrayscaleObservation

env = GrayscaleObservation(env, keep_dim=False)
```

`GrayscaleObservation`を通すと、画像の形は`(84, 84, 3)`から`(84, 84)`へ変わります。

`keep_dim=False`は、色を表していた最後の次元を残さない設定です。このあと直近4枚の画面を重ねると、その4枚が先頭の次元になります。そのため、ここで大きさ`1`の次元を残しておく必要はありません。

## 4. 直近の画面を4枚重ねる

1枚の画像だけでは、マリオが上へ跳んでいるのか、下へ落ちているのか判断できません。写真を1枚見ただけでは、その人がエレベーターで上がっているのか下がっているのか分からないのと同じです。

そこで、直近4枚の画面を1つにまとめます。画面の変化を続けて見れば、マリオや敵がどちらへ動いているか判断しやすくなります。

```python
from gymnasium.wrappers import FrameStackObservation

env = FrameStackObservation(env, stack_size=4)
```

`stack_size=4`は、重ねる画面の枚数です。これで画像の形は`(84, 84)`から`(4, 84, 84)`になります。先頭の`4`には、直近4枚の画面が入っています。

Frame Skipを先に入れたため、ここで重ねるのは4フレームおきの画面です。生の連続4フレームより変化が大きくなり、マリオが上昇中なのか下降中なのかを見分けやすくなります。

## 5. PyTorchのTensorへ変換する

ここまでの`observation`はNumPy配列です。一方、これから作るPyTorchのニューラルネットワークは、**Tensor（テンソル）**という形式でデータを受け取ります。

```python
import torch

observation = torch.as_tensor(observation, dtype=torch.float32) / 255.0
```

`torch.as_tensor()`で、NumPy配列をTensorへ変換します。`dtype=torch.float32`は、ピクセルの値をニューラルネットワークで計算できる小数にする指定です。

元のピクセルは`0`から`255`までの整数なので、最後に`255.0`で割り、`0.0`から`1.0`の範囲へそろえます。数字の大きさをそろえておくと、ニューラルネットワークが学習しやすくなります。

:::note info
Gymnasiumには`NumpyToTorch`というラッパーもあります。ただし、これは画像だけでなく行動も変換します。今回使っている`JoypadSpace`とは行動の形式が合わないため、画像だけを`torch.as_tensor()`で変換します。
:::

## 6. 前処理をまとめる

ここまでのラッパーを、1つのコードにまとめます。第1章と同じ7種類の行動を使うため、`JoypadSpace`も忘れずに追加します。

```python:main.py
import torch

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


env = gym.make(
    "SuperMarioBros-1-1-v0",
    render_mode="rgb_array",
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 同じ行動を4フレーム続ける。
env = SkipFrame(env, skip=4)

# 画面を小さくし、色の情報を減らす。
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env, keep_dim=False)

# 直近4枚の画面を1つの状態にまとめる。
env = FrameStackObservation(env, stack_size=4)

observation, info = env.reset()

# Tensorへ変換し、ピクセルの値を0〜1にそろえる。
observation = torch.as_tensor(observation, dtype=torch.float32) / 255.0

print(type(observation))
print(observation.shape)
print(observation.dtype)
print(env.action_space)

env.close()
```

実行結果は次のようになります。

```text
<class 'torch.Tensor'>
torch.Size([4, 84, 84])
torch.float32
Discrete(7)
```

`reset()`だけでなく、`step()`から新しい画面を受け取ったときも同じ変換を行います。次章では、この処理を関数にまとめて繰り返し使います。

## 第3章のまとめ

この章では、マリオの画面をDQNへ渡す準備をしました。

1. 同じ行動を4フレーム続け、その間の報酬を合計した
2. 画面を`84 × 84`へ縮小した
3. カラー画像をグレースケールにした
4. 4フレームおきの画面を4枚重ね、動きが分かる状態にした
5. NumPy配列をTensorへ変換し、ピクセルの値を`0〜1`にそろえた

最終的な`observation`の形は`(4, 84, 84)`です。これで、直近の動きを含んだ画面をPyTorchで扱えるようになりました。

材料の下ごしらえは完了です。次の章では、この画面を受け取り、7種類の行動それぞれのQ値を予測するQ-Networkを作ります。

# 第4章 Q-Networkに画面を見せよう

第2章では、「Qテーブルの代わりにニューラルネットワークでQ値を予測する」と説明しました。第3章では、その入力となるゲーム画面も準備しました。いよいよ、この2つをつなぎます。

この章で作るのは、ゲーム画面を受け取り、7種類の行動それぞれのQ値を出力する**Q-Network**です。

```text
ゲーム画面
    ↓
Q-Network
    ↓
7種類の行動のQ値
```

ただし、この章ではまだ学習させません。まずは画面を入れると7個の数字が出てくる、Q-Networkの配管工事から始めます。賢くなるのはもう少し先です。

## 1. 画像からQ値が出るまで

前章で作った状態の形は`(4, 84, 84)`でした。直近4枚のグレースケール画像が重なっています。

Q-Networkでは、まず**CNN（畳み込みニューラルネットワーク）**を使って画像の特徴を探します。CNNは小さな窓を画像の上で動かしながら、線や角、形など、判断の手がかりになりそうなものを見つける仕組みです。学習が進めば、その手がかりからマリオや敵、足場の位置を捉えられるようになります。

CNNが見つけた特徴は、**全結合層**へ渡します。全結合層はそれらをまとめ、「この画面なら右は何点、ジャンプは何点」と7個のQ値に変換します。

```text
ゲーム画面 (4, 84, 84)
        ↓
CNNで画像の特徴を探す
        ↓
全結合層で特徴をまとめる
        ↓
7個のQ値
```

## 2. 実装

それでは、Q-NetworkをPyTorchで実装しましょう。PyTorchでは、`nn.Module`を継承してニューラルネットワークのクラスを作ります。

```python
from torch import nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
```

`nn.Module`には、ニューラルネットワークの層や学習中の値を管理する機能が用意されています。`super().__init__()`は、その機能を使うための初期設定です。ここは定型文として覚えておけば大丈夫です。

### 層を順番に並べる

次に、画像が通る層を`nn.Sequential`で上から順番に並べます。

```python
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)
```

それぞれの層の役割を、ざっくり確認しておきましょう。

| 層 | 仕事 |
| --- | --- |
| `Conv2d` | 画像の一部分ずつを見て、特徴を探す |
| `ReLU` | マイナスの値を`0`にして、特徴を扱いやすくする |
| `Flatten` | 縦・横に並んだ特徴を、1列に並べ直す |
| `Linear` | 見つけた特徴を組み合わせ、最後にQ値を出す |

`Conv2d`に指定している数字も見てみましょう。

```python
nn.Conv2d(4, 32, kernel_size=8, stride=4)
```

- 最初の`4`：入力する直近4枚の画面
- `32`：この層が作る特徴の数
- `kernel_size=8`：一度に見る窓の大きさ
- `stride=4`：その窓を一度に動かす距離

虫眼鏡を画面の左上から少しずつ動かす様子を想像すると分かりやすいでしょう。`kernel_size`が虫眼鏡の大きさ、`stride`が一歩の大きさです。

画像が3つの`Conv2d`を通ると、その形は次のように変わります。

| 場所 | データの形 |
| --- | --- |
| 入力 | `(4, 84, 84)` |
| 1つ目の`Conv2d`の後 | `(32, 20, 20)` |
| 2つ目の`Conv2d`の後 | `(64, 9, 9)` |
| 3つ目の`Conv2d`の後 | `(64, 7, 7)` |

そのため、`Flatten`で1列に並べると要素数は`64 × 7 × 7 = 3136`になります。これが`nn.Linear(7 * 7 * 64, 512)`の`7 * 7 * 64`の正体です。計算方法を暗記する必要はありません。「画像が層を通るたびに小さくなり、最後に1列へ並ぶ」と分かれば十分です。

最後の`nn.Linear(512, n_actions)`が、まとめた特徴を行動ごとのQ値へ変換します。`n_actions`には、マリオが選べる行動数の`7`が入ります。

### `forward`でデータの通り道を決める

`forward()`は、受け取ったデータをどの層へ通すかを決めるメソッドです。今回は`nn.Sequential`へすべて並べたので、`self.net(x)`だけで済みます。

```python
q_network = QNetwork(n_actions=7)
q_values = q_network(state)
```

`q_network(state)`と呼ぶと、PyTorchが内部で`forward(state)`を実行します。`forward()`を直接呼ばず、モデルそのものを関数のように呼ぶのがPyTorchの基本的な使い方です。

## 3. 実際に動かしてみる

前章の前処理と、今作ったQ-Networkをつないでみましょう。完成したコードは次のとおりです。

```python:main.py
import torch
from torch import nn

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def make_env(render_mode="rgb_array"):
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render_mode,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def to_tensor(observation):
    observation = torch.as_tensor(observation, dtype=torch.float32) / 255.0
    return observation.unsqueeze(0)


def main():
    env = make_env()
    observation, info = env.reset()

    state = to_tensor(observation)
    q_network = QNetwork(n_actions=env.action_space.n)

    with torch.no_grad():
        q_values = q_network(state)

    action = q_values.argmax(dim=1).item()

    print(f"入力の形: {state.shape}")
    print(f"出力の形: {q_values.shape}")
    print(f"Q値: {q_values}")
    print(f"選んだ行動: {action} {SIMPLE_MOVEMENT[action]}")

    env.close()


if __name__ == "__main__":
    main()
```

### バッチの次元を追加する

`to_tensor()`では、前章と同じ変換に加えて、最後に`unsqueeze(0)`を呼んでいます。

```python
return observation.unsqueeze(0)
```

PyTorchの`Conv2d`は、複数の画像をまとめて処理できるように、入力を`(個数, チャンネル, 高さ, 幅)`の形で受け取ります。今回は状態が1個だけなので、先頭に`1`を追加します。

```text
(4, 84, 84)
    ↓ unsqueeze(0)
(1, 4, 84, 84)
```

この先頭の「まとめて処理する個数」を**バッチサイズ**と呼びます。今は1個だけですが、学習するときには複数の経験をまとめて処理します。

### 一番高いQ値の行動を選ぶ

`q_network(state)`を実行すると、7種類の行動に対応するQ値が返ってきます。

```python
with torch.no_grad():
    q_values = q_network(state)
```

`torch.no_grad()`は、「今は動作確認だけなので、学習に使う計算記録は残さなくてよい」という指定です。

続いて、Q値が一番高い行動の番号を取得します。

```python
action = q_values.argmax(dim=1).item()
```

`argmax()`は最大値そのものではなく、最大値が入っている場所を返します。最後の`.item()`で、`JoypadSpace`へ渡せるPythonの整数に変換しています。

実行すると、次のように表示されます。

```text
入力の形: torch.Size([1, 4, 84, 84])
出力の形: torch.Size([1, 7])
Q値: tensor([[ 0.01, -0.02, 0.03, ... ]])
選んだ行動: 2 ['right', 'A']
```

Q値や選ばれる行動は、実行するたびに変わることがあります。今のQ-Networkは作ったばかりで、まだ一度も学習していないからです。現時点では、7個の数字をそれらしい顔で出しているだけ。マリオより先にネットワークが迷子です。

それでも、ゲーム画面を入力し、7個のQ値を出し、一番高い行動を選ぶところまでつながりました。

## 4. 推論でマリオを動かしてみる

ここまでは、最初の画面から行動を1回選んだだけでした。今度は同じ処理を繰り返し、Q-Networkにマリオを操作させてみましょう。

Q-Networkへ状態を入力し、出力されたQ値から行動を決める処理を **推論（inference）** と呼びます。

```text
現在の画面を受け取る
        ↓
Q-NetworkでQ値を予測する
        ↓
一番高いQ値の行動を選ぶ
        ↓
その行動でゲームを最大4フレーム進める
        ↓
次の画面で繰り返す
```

学習ではQ-Networkの中身を書き換えますが、推論では現在のQ-Networkを使って行動を選ぶだけです。

### 推論用のコードを作る

`04-Q-network`フォルダへ、`inference.py`を追加します。先ほど`main.py`に作った`QNetwork`、`make_env()`、`to_tensor()`は、そのまま再利用できます。

```python:inference.py
import time

import torch

from main import QNetwork, make_env, to_tensor


def play(env, q_network, max_steps=1000):
    observation, info = env.reset()
    q_network.eval()

    try:
        for _ in range(max_steps):
            state = to_tensor(observation)

            with torch.no_grad():
                q_values = q_network(state)

            action = q_values.argmax(dim=1).item()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(4 / 60)

            if terminated or truncated:
                observation, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    env = make_env(render_mode="human")
    q_network = QNetwork(n_actions=env.action_space.n)
    play(env, q_network)


if __name__ == "__main__":
    main()
```

SkipFrameによって`env.step()`を1回呼ぶと最大4フレーム進むため、表示時も`time.sleep(4 / 60)`で約4フレーム分待ちます。

推論ではゲーム画面を見たいので、`make_env(render_mode="human")`で環境を作ります。これに合わせて、`main.py`の`make_env()`も表示方法を受け取れるように変更しておきます。

```python
def make_env(render_mode="rgb_array"):
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render_mode,
    )
    # 以下は同じ
```

### 推論用のモードへ切り替える

```python
q_network.eval()
```

`eval()`は、Q-Networkを推論用の動作へ切り替えます。現在のQ-Networkには、学習中と推論中で動きが変わる層は入っていませんが、推論するときに呼ぶ習慣をつけておきましょう。

さらに、Q値の計算を`torch.no_grad()`で囲みます。

```python
with torch.no_grad():
    q_values = q_network(state)
```

推論ではQ-Networkを更新しないため、学習用の計算記録は必要ありません。`torch.no_grad()`を使うと、その記録を作らずにQ値だけを計算できます。

:::note info
`eval()`はモデルの動作モードを切り替え、`torch.no_grad()`は学習用の計算記録を止めます。似ていますが役割は別なので、推論時は両方使います。
:::

### 実行してみる

`04-Q-network`フォルダで次のコマンドを実行します。

```bash
uv run python inference.py
```

ウィンドウが開き、Q-Networkが選んだ行動でマリオが動きます。`Ctrl + C`で終了できます。

ただし、今のQ-Networkはまだ学習していません。最初にたまたま決まったQ値を頼りに動くため、その場から動かなかったり、左へ進もうとしたり、同じ行動を繰り返したりします。自由というより迷走です。

それでも、推論のコード自体はこれで完成です。学習後も使う流れは同じで、変わるのはQ-Networkの中身だけです。あとで学習済みの値を読み込めば、この同じ`play()`で成長したマリオを動かせます。

## 第4章のまとめ

この章では、Q-Networkの入口から出口までを作りました。

1. CNNでゲーム画面の特徴を探す
2. 全結合層で特徴を7個のQ値へ変換する
3. 状態にバッチの次元を加え、Q-Networkへ入力する
4. 一番高いQ値を持つ行動を選ぶ
5. 推論ループを使い、Q-Networkにマリオを操作させる

推論の流れは完成しましたが、出力されたQ値にはまだ意味がありません。Q-Networkを賢くするには、マリオが実際に行動し、その結果を経験として集める必要があります。次の章では、その経験を保存する仕組みを作ります。

# 第5章 マリオの経験をためよう

前章では、ゲーム画面から7個のQ値を出力できました。しかし、できたばかりのQ-Networkは、まだ何がよい行動なのか知りません。ジャンプがクリボーを避ける行動なのか、穴へ飛び込む行動なのかも分からない状態です。

Q値を学ぶには、マリオが実際に行動し、その結果を集める必要があります。

```text
現在の状態で
    ↓
行動を選び
    ↓
報酬と次の状態を受け取る
```

この一連の出来事を、ここでは**経験（experience）**と呼びます。この章では、マリオの経験を保存し、あとからランダムに取り出せるようにします。

## 1. 1つの経験に何を保存するか

1つの経験には、次の5つを保存します。

| 名前 | 保存するもの |
| --- | --- |
| `state` | 行動する前の状態 |
| `action` | 選んだ行動 |
| `reward` | 行動して受け取った報酬 |
| `next_state` | 行動した後の状態 |
| `done` | その行動でゲームが終了したか |

例えば、「クリボーが目の前にいる状態でジャンプしたら、クリボーを越えて先へ進めた」という出来事を、5つのデータに分けて保存するイメージです。

```text
クリボーが目の前にいる画面  → state
ジャンプ                      → action
進んだことで得た報酬          → reward
クリボーを越えた後の画面      → next_state
ゲームは続いている            → done = False
```

Pythonでは、これらを`dataclass`にまとめます。

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
```

`dataclass`を使うと、関連するデータを1つのまとまりとして扱えます。例えば、保存した経験が`experience`に入っていれば、`experience.action`や`experience.reward`のように取り出せます。

状態はTensorではなく、環境から受け取ったNumPy配列のまま保存します。元の`uint8`形式なら、各ピクセルを`0〜255`の整数1つで持てるため、`float32`のTensorより少ない容量で済みます。学習するときに、取り出した状態だけをまとめてTensorへ変換しましょう。

## 2. マリオはどんな報酬を受け取るのか

経験に保存する5つのデータのうち、`reward`についてもう少し見ておきましょう。

第2章では、報酬を「よい行動でもらえるご褒美、悪い行動でもらう罰」と説明しました。では、マリオの何がよくて、何が悪いのでしょうか。

実は、採点はすでにゲーム環境の中で行われています。`env.step()`で行動すると、環境は画面を進めるだけでなく、その結果を採点して`reward`として返してくれます。

```python
next_state, reward, terminated, truncated, info = env.step(action)
```

このチュートリアルで使っている環境では、主に次の変化から報酬が作られます。

| 項目 | どんなときに変化するか |
| --- | --- |
| `progress` | これまでの最高地点より先へ進んだ |
| `time` | 残り時間が減った |
| `score` | ゲーム内のスコアが増えた |
| `coins` | コインを取った |
| `powerup` | キノコなどでパワーアップした |
| `completion` | ステージをクリアした |
| `death` | マリオがやられた |

例えば右へ進んだときは、進んだ距離に応じて`progress`がプラスになります。一方、時間が減ると`time`がマイナスになり、マリオがやられると`death`がマイナスになります。道草より前進、生存よりクリア。なかなかせっかちな採点係です。

1フレーム分の内訳は、`info["reward_components"]`で確認できます。

```python
print(f"今回の報酬: {reward}")
print(info["reward_components"])
```

SkipFrameは4フレーム分の報酬を合計して返します。一方、`info`に入っているのは、そのうち最後に進めたフレームの情報です。そのため、`reward`と`reward_components`の合計が一致しない場合もあります。

例えば、最後のフレームで右へ少し進んでいれば、`reward_components`は次のようになります。

```text
今回の報酬: 2.0
{'progress': 1.0, 'time': 0.0, 'score': 0.0, 'coins': 0.0, 'powerup': 0.0, 'completion': 0.0, 'death': 0.0}
```

この例では、4フレーム分の報酬は`2.0`ですが、最後のフレームだけを見ると、前へ進んだことで`progress`が`1.0`になっています。複数の出来事が同時に起きれば、それぞれの値を合わせて1フレーム分の報酬が決まります。ゲーム環境は1フレームの報酬を`-15〜15`の範囲に収め、その最大4フレーム分をSkipFrameが合計します。

この「何をしたら何点にするか」という採点方法を、**報酬設計**と呼びます。エージェントは、こちらの気持ちを察して動いているわけではありません。報酬をできるだけ多く集めようとしているだけです。

そのため、採点基準にうまく表れていない行動を覚えることもあります。例えば、その場でジャンプし続けることには専用の減点がありません。こちらから見れば「前へ進んでくれ」と言いたくなりますが、マリオにはそのため息が聞こえないのです。

まずは、環境から受け取った報酬をそのまま使って学習させます。その結果を見て困った行動が見つかったら、あとで採点方法を少し調整してみましょう。

## 3. なぜ経験をためるのか

経験を受け取るたび、すぐQ-Networkへ学習させることもできそうです。しかし、Frame Skipで4フレームおきにしても、前後のゲーム画面はまだよく似ています。

```text
現在の画面
次の画面       ← ほとんど同じ
さらに次の画面 ← やっぱりよく似ている
```

似た経験ばかりを続けて見せると、Q-Networkの学習が一部の場面へ偏ってしまいます。

そこで、経験をいったん箱へためておき、学習するときにランダムに取り出します。クリボーの前、土管の前、何もない道など、時間の離れた経験を混ぜて復習できるわけです。単語帳を上から順番に丸暗記するのではなく、カードをシャッフルして解くようなものですね。

この経験をためる箱を**Replay Memory（リプレイメモリ）**と呼びます。

## 4. Replay Memoryを作る

Replay Memoryには、次の3つの機能を持たせます。

1. 新しい経験を保存する
2. 保存した経験からランダムに取り出す
3. 現在いくつ保存されているかを返す

```python
import random
from collections import deque


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

### 古い経験から忘れる

`deque`は、データを順番に保存できる入れ物です。`maxlen=capacity`で、保存できる最大数を決めています。

容量が1000のReplay Memoryへ1001個目の経験を追加すると、一番古い経験が自動的に削除されます。すべてを覚え続けるとメモリがいっぱいになるので、古い思い出からそっと忘れてもらいます。マリオにも記憶容量の限界はあります。

### 経験をランダムに取り出す

`sample()`では、`random.sample()`を使って経験をランダムに取り出します。

```python
experiences = memory.sample(batch_size=4)
```

この場合は4個の経験が返ります。このように、学習のために一度に取り出すデータのまとまりを、前章で登場した**バッチ**と呼びます。

:::note info
保存されている経験より大きな`batch_size`は指定できません。例えば4個取り出したいなら、先に4個以上の経験をためておく必要があります。
:::

## 5. マリオの経験を集める

Replay Memoryができたので、実際にマリオを動かして経験を集めます。

```python
state, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    memory.push(
        state=state.copy(),
        action=int(action),
        reward=float(reward),
        next_state=next_state.copy(),
        done=done,
    )

    if done:
        state, info = env.reset()
    else:
        state = next_state
```

今のQ-Networkは未学習なので、今回は`env.action_space.sample()`で行動をランダムに選びます。上手には動けませんが、まずはさまざまな行動の結果を集めることが目的です。

`state.copy()`と`next_state.copy()`では、その時点の画像をコピーして保存しています。あとから画像の中身が変わっても、過去の経験まで書き換わらないようにするためです。

`done`は、`terminated`または`truncated`のどちらかが`True`なら`True`になります。ゲームが終了した場合は`reset()`し、新しいプレイの最初から経験集めを再開します。

## 6. 完成したコード

ここまでをまとめると、次のようになります。

```python:main.py
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def make_env():
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def main():
    env = make_env()
    memory = ReplayMemory(capacity=1000)
    state, info = env.reset()

    for _ in range(500):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.push(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            done=done,
        )

        if done:
            state, info = env.reset()
        else:
            state = next_state

    experiences = memory.sample(batch_size=4)

    print(f"保存した経験: {len(memory)}個")
    print(f"取り出した経験: {len(experiences)}個")
    print(f"状態の形: {experiences[0].state.shape}")
    print(f"行動: {experiences[0].action}")
    print(f"報酬: {experiences[0].reward}")
    print(f"終了したか: {experiences[0].done}")

    env.close()


if __name__ == "__main__":
    main()
```

実行すると、500個の経験を保存したあと、その中から4個をランダムに取り出します。

```text
保存した経験: 500個
取り出した経験: 4個
状態の形: (4, 84, 84)
行動: 5
報酬: 3.0
終了したか: False
```

行動や報酬はランダムに選ばれた経験なので、実行するたびに変わります。

## 第5章のまとめ

この章では、マリオの経験を保存するReplay Memoryを作りました。

1. 状態・行動・報酬・次の状態・終了フラグを1つの経験にまとめた
2. ゲーム環境がどのようにマリオの行動を採点しているか確認した
3. `deque`を使い、決めた数だけ経験を保存できるようにした
4. 保存した経験から、バッチをランダムに取り出せるようにした
5. ランダムにマリオを動かし、実際の経験を集めた

これで、Q-Networkに見せるための教材がそろいました。次の章ではReplay Memoryから経験を取り出し、第2章で登場したQ学習の式を使って、いよいよQ-Networkの点数を更新します。

<!-- 第5章ドラフトここまで -->

# 第6章 Q-Networkを学習させよう

第5章では、マリオの経験をReplay Memoryへ保存しました。教材はそろいました。今度は、その経験を使ってQ-NetworkのQ値を更新します。

学習の流れは、テストの答え合わせによく似ています。

```text
Q-Networkが予測したQ値
        ↓ 比べる
経験から作った目標のQ値
        ↓
どれくらい外れたかを計算する
        ↓
Q-Networkを少し修正する
```

この章では、Replay Memoryから1バッチを取り出し、Q-Networkを1回更新するところまで進めます。ついに「学習」と呼べる処理が始まります。長い準備運動でした。マリオならもうゴールしていそうです。

## 1. 予測したQ値と目標のQ値

Q-Networkを学習させるには、2種類のQ値を用意します。

- **予測したQ値**：現在のQ-Networkが出した点数
- **目標のQ値**：報酬と次の状態から作った、目指してほしい点数

例えば、「右へ進む」を選んだとき、Q-Networkが`0.5`と予測したとします。しかし、実際には前へ進んで報酬を受け取り、その先にもよい行動がありました。経験から計算した目標が`1.2`なら、予測は少し低すぎます。

```text
予測したQ値: 0.5
目標のQ値  : 1.2
        ↓
もう少し高くなるように修正する
```

反対に、予測が目標より高すぎれば、Q値が下がるように修正します。この答え合わせを繰り返すことで、Q-Networkの予測が少しずつ経験に近づいていきます。

## 2. 経験をバッチへまとめる

まず、Replay Memoryから経験を32個取り出します。

```python
BATCH_SIZE = 32

experiences = memory.sample(BATCH_SIZE)
```

それぞれの経験には、`state`、`action`、`reward`、`next_state`、`done`が入っています。同じ種類のデータをまとめ、PyTorchのTensorへ変換します。

```python
def states_to_tensor(states):
    states = np.stack(states)
    return torch.as_tensor(states, dtype=torch.float32) / 255.0


states = states_to_tensor(
    [experience.state for experience in experiences]
)

actions = torch.tensor(
    [experience.action for experience in experiences],
    dtype=torch.long,
).unsqueeze(1)

rewards = torch.tensor(
    [experience.reward for experience in experiences],
    dtype=torch.float32,
)

next_states = states_to_tensor(
    [experience.next_state for experience in experiences]
)

dones = torch.tensor(
    [experience.done for experience in experiences],
    dtype=torch.float32,
)
```

`np.stack()`は、32個の状態を先頭方向へ積み重ねます。1つの状態は`(4, 84, 84)`なので、まとめた後の形は`(32, 4, 84, 84)`です。

| Tensor | 形 | 中身 |
| --- | --- | --- |
| `states` | `(32, 4, 84, 84)` | 行動前の状態 |
| `actions` | `(32, 1)` | 選んだ行動番号 |
| `rewards` | `(32,)` | 受け取った報酬 |
| `next_states` | `(32, 4, 84, 84)` | 行動後の状態 |
| `dones` | `(32,)` | ゲームが終了したか |

第4章では、バッチサイズが`1`でした。今回は32個の経験をまとめて学習するので、先頭が`32`になっています。

## 3. 選んだ行動のQ値を取り出す

`states`をQ-Networkへ入力すると、32個の状態それぞれに7個のQ値が返ります。

```python
all_q_values = q_network(states)
```

`all_q_values`の形は`(32, 7)`です。しかし、経験の中で実際に選んだ行動は、それぞれ1つだけです。学習には、その行動に対応するQ値を使います。

例えば、次の2つの経験があったとします。

```text
1つ目のQ値: [0.1, 0.2, 0.8, 0.3]  選んだ行動: 2
2つ目のQ値: [0.6, 0.1, 0.2, 0.4]  選んだ行動: 0
```

取り出したいのは、1つ目の`0.8`と、2つ目の`0.6`です。PyTorchでは`gather()`を使って取り出せます。

```python
predicted_q_values = q_network(states).gather(1, actions).squeeze(1)
```

これが、Q-Networkが現在予測しているQ値です。形は`(32,)`になり、経験1つにつき1個のQ値が並びます。

## 4. 目標のQ値を作る

次は、予測が目指す目標のQ値を作ります。ここで、第2章のQ学習が戻ってきます。

```text
目標のQ値
= 今回の報酬
  + 割引率 × 次の状態で一番高いQ値
```

コードでは次のようになります。

```python
GAMMA = 0.99

with torch.no_grad():
    next_q_values = target_network(next_states).max(dim=1).values
    target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)
```

`max(dim=1).values`で、次の状態にある7個のQ値から一番高いものを選びます。

`GAMMA`は、第2章で登場した割引率です。第2章では計算を追いやすくするため`0.9`を使いましたが、ここでは遠くの報酬も残りやすい`0.99`を使います。どちらも絶対の正解ではなく、どれくらい先の報酬を大切にするかを決める設定です。

式の最後にある`(1.0 - dones)`にも意味があります。

- ゲームが続くとき：`done = 0`なので、次の状態のQ値を加える
- ゲームが終わったとき：`done = 1`なので、次の状態のQ値を`0`にする

ゲームオーバーの先には、次の行動がありません。そのため、終了した経験の目標は、最後に受け取った報酬だけになります。

### Target Networkとは

目標のQ値を計算するときは、学習中のQ-Networkとは別に、**Target Network**を使います。

```python
q_network = QNetwork(n_actions=env.action_space.n)
target_network = QNetwork(n_actions=env.action_space.n)

target_network.load_state_dict(q_network.state_dict())
target_network.eval()
```

最初は、Q-Networkの内容をTarget Networkへそのままコピーします。

もし学習中のQ-Networkだけで予測と目標の両方を作ると、予測を直すたびに目標まで動いてしまいます。採点中に模範解答まで書き換わるようなもので、なかなか答えに近づけません。

そこで、目標を作るTarget Networkはしばらく固定しておきます。何回か学習したら、最新のQ-Networkを再びコピーします。今回は1回だけ学習するので、最初にコピーした状態のまま使います。

`torch.no_grad()`で囲んでいるのも、Target Networkは今回更新しないためです。「目標を計算するだけで、こちらは直さない」とPyTorchへ伝えています。

## 5. 誤差を使ってQ-Networkを更新する

予測したQ値と目標のQ値がそろったので、どれくらい離れているかを計算します。このズレを**損失（loss）**と呼びます。

```python
loss = nn.SmoothL1Loss()(predicted_q_values, target_q_values)
```

損失が小さいほど、Q-Networkの予測が目標に近いことを表します。`SmoothL1Loss`は、予測が大きく外れた経験があっても、その1件に振り回されすぎないように比べてくれる関数です。

続いて、損失をもとにQ-Networkを更新します。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

3行の役割は次のとおりです。

1. `zero_grad()`：前回の計算結果をリセットする
2. `backward()`：どの値をどちらへ直せばよいか調べる
3. `step()`：Q-Networkの値を実際に少し動かす

更新には`Adam`というOptimizerを使います。Optimizerは、`backward()`で調べた結果をもとに、ネットワークをどれくらい動かすか決める係です。

```python
LEARNING_RATE = 0.0001

optimizer = torch.optim.Adam(
    q_network.parameters(),
    lr=LEARNING_RATE,
)
```

`lr`は学習率です。大きすぎると点数を直しすぎ、小さすぎるとなかなか変わりません。今回は一度に少しずつ直すため、`0.0001`にします。

## 6. 1回の学習処理をまとめる

ここまでの処理を`train_step()`へまとめます。

```python
def train_step(q_network, target_network, memory, optimizer):
    experiences = memory.sample(BATCH_SIZE)

    states = states_to_tensor([experience.state for experience in experiences])
    actions = torch.tensor(
        [experience.action for experience in experiences],
        dtype=torch.long,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [experience.reward for experience in experiences],
        dtype=torch.float32,
    )
    next_states = states_to_tensor(
        [experience.next_state for experience in experiences]
    )
    dones = torch.tensor(
        [experience.done for experience in experiences],
        dtype=torch.float32,
    )

    predicted_q_values = q_network(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)

    loss = nn.SmoothL1Loss()(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

`loss.item()`は、Tensorになっている損失からPythonの数値だけを取り出します。これで、学習中の損失を画面へ表示したり、あとからグラフにしたりできます。

## 7. 完成したコード

これまでに作ったQ-NetworkとReplay Memoryを組み合わせると、コード全体は次のようになります。

```python:main.py
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from nes_py.wrappers import JoypadSpace


BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.0001


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return observation, total_reward, terminated, truncated, info


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def make_env():
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode="rgb_array",
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)
    return env


def states_to_tensor(states):
    states = np.stack(states)
    return torch.as_tensor(states, dtype=torch.float32) / 255.0


def train_step(q_network, target_network, memory, optimizer):
    experiences = memory.sample(BATCH_SIZE)

    states = states_to_tensor([experience.state for experience in experiences])
    actions = torch.tensor(
        [experience.action for experience in experiences],
        dtype=torch.long,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [experience.reward for experience in experiences],
        dtype=torch.float32,
    )
    next_states = states_to_tensor(
        [experience.next_state for experience in experiences]
    )
    dones = torch.tensor(
        [experience.done for experience in experiences],
        dtype=torch.float32,
    )

    predicted_q_values = q_network(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * next_q_values * (1.0 - dones)

    loss = nn.SmoothL1Loss()(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def collect_experiences(env, memory, count):
    state, info = env.reset()

    for _ in range(count):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.push(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            done=done,
        )

        if done:
            state, info = env.reset()
        else:
            state = next_state


def main():
    env = make_env()
    memory = ReplayMemory(capacity=1000)

    q_network = QNetwork(n_actions=env.action_space.n)
    target_network = QNetwork(n_actions=env.action_space.n)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(
        q_network.parameters(),
        lr=LEARNING_RATE,
    )

    collect_experiences(env, memory, count=500)
    loss = train_step(q_network, target_network, memory, optimizer)

    print(f"保存した経験: {len(memory)}個")
    print(f"バッチサイズ: {BATCH_SIZE}")
    print(f"損失: {loss:.4f}")

    env.close()


if __name__ == "__main__":
    main()
```

実行すると、500個の経験を集め、その中から32個を使ってQ-Networkを1回更新します。

```text
保存した経験: 500個
バッチサイズ: 32
損失: 0.9552
```

損失は、集めた経験やQ-Networkの最初の状態によって毎回変わります。また、1回の学習で損失が小さくなったからといって、すぐマリオが上手になるわけではありません。今は、学習処理が最初から最後までつながったことが大切です。

## 第6章のまとめ

この章では、Replay Memoryの経験を使い、Q-Networkを1回更新しました。

1. 32個の経験をバッチへまとめた
2. Q-Networkから、実際に選んだ行動のQ値を取り出した
3. 報酬と次の状態から、目標のQ値を作った
4. Target Networkを使い、目標がすぐ動かないようにした
5. 予測と目標の損失を計算し、Q-Networkを更新した

これで、DQNに必要な部品はひととおりそろいました。次の章では、行動の選択、経験の保存、Q-Networkの更新、Target Networkの同期を1つのループへまとめ、マリオを繰り返し学習させます。

<!-- 第6章ドラフトここまで -->

# 第7章 マリオを繰り返し学習させよう

これまでに、DQNを作るための部品を1つずつ用意してきました。

- Q-Networkで行動ごとのQ値を予測する
- Replay Memoryへ経験を保存する
- 経験を使ってQ-Networkを更新する
- Target Networkで目標のQ値を作る

あとは、これらを繰り返すだけです。この章では、行動の選択から学習、モデルの保存までを1本のループへまとめます。

```text
行動を選ぶ
    ↓
ゲームを進める
    ↓
経験を保存する
    ↓
Q-Networkを学習させる
    ↓
Target Networkを更新する
    ↓
また行動を選ぶ
```

ついにマリオの自主練が始まります。人間は見守る係です。

## 1. 学習の設定を決める

まずは、学習に使う設定をまとめておきます。

```python
TOTAL_STEPS = 100_000
MEMORY_CAPACITY = 5000
WARMUP_STEPS = 1000
BATCH_SIZE = 32
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 1000
SAVE_INTERVAL = 10_000

GAMMA = 0.99
LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 50_000
```

名前が多いですが、一度に覚える必要はありません。役割は次のとおりです。

| 名前 | 役割 |
| --- | --- |
| `TOTAL_STEPS` | 行動を選ぶ合計ステップ数 |
| `MEMORY_CAPACITY` | 保存する経験の最大数 |
| `WARMUP_STEPS` | 学習を始める前に集める経験の数 |
| `BATCH_SIZE` | 1回の学習に使う経験の数 |
| `TRAIN_INTERVAL` | 何ステップごとに学習するか |
| `TARGET_UPDATE_INTERVAL` | Target Networkを更新する間隔 |
| `SAVE_INTERVAL` | モデルを途中保存する間隔 |

SkipFrameを使っているため、ここでいう1ステップではゲームが最大4フレーム進みます。`100_000`ステップは、コードが長時間動くことを確認するための出発点です。この回数だけで必ず1-1をクリアできるわけではありません。強化学習は、思っているより気が長い世界です。

:::note warning
`MEMORY_CAPACITY = 5000`では、Replay Memoryが数百MBほど使うことがあります。メモリに余裕がなければ、まず`1000`へ減らして動作を確認してください。
:::

## 2. 計算に使うデバイスを選ぶ

学習は計算量が多いため、利用できる場合はGPUを使います。NVIDIAのGPU、Apple SiliconのGPU、CPUの順に確認します。

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- `cuda`：NVIDIAのGPU
- `mps`：Apple SiliconのGPU
- `cpu`：CPU

選んだデバイスへ、2つのネットワークを移動します。

```python
device = get_device()

q_network = QNetwork(env.action_space.n).to(device)
target_network = QNetwork(env.action_space.n).to(device)
```

バッチをTensorへ変換するときも、同じデバイスを指定します。ネットワークとデータが別々の場所にいると計算できないためです。遠距離恋愛には厳しいPyTorchです。

```python
def states_to_tensor(states, device):
    states = np.stack(states)
    return torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    ) / 255.0
```

## 3. ε-greedyで行動を選ぶ

第2章で登場したε-greedyを、実際のコードにします。

学習を始めた直後は、Q-Networkの予測を信用できません。そこで、最初はほとんどランダムに行動します。経験が増えるにつれてランダム行動を減らし、Q値の高い行動を使う割合を増やします。

```text
学習開始時  ε = 1.0  → ほぼすべてランダム
学習途中    ε = 0.5  → 半分くらいランダム
学習後半    ε = 0.1  → たまにランダム
```

現在のステップからεを計算する関数を作ります。

```python
def calculate_epsilon(step):
    progress = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)
```

今回は、最初の`50,000`ステップを使って、εを`1.0`から`0.1`まで少しずつ下げます。それ以降は`0.1`のままです。学習後半にも少しだけランダム行動を残し、知らない動きを試せるようにします。

行動を選ぶ処理は次のようになります。

```python
def select_action(state, q_network, env, epsilon, device):
    if random.random() < epsilon:
        return int(env.action_space.sample())

    state_tensor = states_to_tensor([state], device)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    return q_values.argmax(dim=1).item()
```

`random.random()`は、`0.0`以上`1.0`未満の数をランダムに返します。その値がεより小さければランダム行動、それ以外ならQ値が一番高い行動を選びます。

## 4. 最初は経験集めに専念する

Replay Memoryが空のままでは、バッチを取り出せません。また、数件だけで学習を始めても、似た経験ばかりになってしまいます。

そこで、最初の`1000`ステップは経験集めに専念します。この期間を**Warm-up（ウォームアップ）**と呼びます。

```python
if len(memory) >= WARMUP_STEPS and step % TRAIN_INTERVAL == 0:
    latest_loss = train_step(
        q_network,
        target_network,
        memory,
        optimizer,
        device,
    )
```

経験が1000個以上たまったら、4ステップごとに`train_step()`を呼びます。毎ステップ学習するより計算量を抑えながら、定期的にQ-Networkを更新できます。

## 5. Target Networkを定期的に同期する

Target Networkは、目標のQ値を安定させるため、しばらく同じ内容で使います。ただし、ずっと最初のままではQ-Networkの成長についていけません。

そこで、1000ステップごとに、最新のQ-NetworkをTarget Networkへコピーします。

```python
if step % TARGET_UPDATE_INTERVAL == 0:
    target_network.load_state_dict(q_network.state_dict())
```

毎回追いかけるのではなく、ときどき最新情報を受け取る。Target Networkは、少し返信の遅い友人くらいがちょうどよいのです。

## 6. 学習したモデルを保存する

学習には時間がかかります。途中でプログラムを終了しても続きが残るように、Q-Networkを定期的に保存します。

```python
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "q_network.pt"
```

```python
if step % SAVE_INTERVAL == 0:
    torch.save(q_network.state_dict(), MODEL_PATH)
```

`state_dict()`には、Q-Networkが学習した値が入っています。`torch.save()`で保存すると、`q_network.pt`というファイルが作られます。

`Ctrl + C`で中断した場合にも保存できるよう、最後の処理は`finally`へ置きます。

```python
finally:
    torch.save(q_network.state_dict(), MODEL_PATH)
    env.close()
```

これで、10,000ステップごとの途中保存に加え、正常終了や中断時にも最新のモデルが残ります。

## 7. 学習ループを組み立てる

ここまでの処理を1つのループへまとめます。

```python
for step in range(1, TOTAL_STEPS + 1):
    epsilon = calculate_epsilon(step)
    action = select_action(state, q_network, env, epsilon, device)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    memory.push(
        state=state.copy(),
        action=action,
        reward=float(reward),
        next_state=next_state.copy(),
        done=done,
    )

    episode_reward += reward

    if len(memory) >= WARMUP_STEPS and step % TRAIN_INTERVAL == 0:
        latest_loss = train_step(
            q_network,
            target_network,
            memory,
            optimizer,
            device,
        )

    if step % TARGET_UPDATE_INTERVAL == 0:
        target_network.load_state_dict(q_network.state_dict())

    if step % SAVE_INTERVAL == 0:
        torch.save(q_network.state_dict(), MODEL_PATH)

    if done:
        state, info = env.reset()
        episode += 1
        episode_reward = 0.0
    else:
        state = next_state
```

ループの中で行っていることは、今まで作った処理ばかりです。新しい大技ではなく、これまでの部品を順番につないだものです。

ゲームが終了したら`reset()`して次のエピソードへ進みます。ゲーム開始から終了までの1回分を**エピソード（episode）**と呼びます。

実際のコードでは、エピソードが終わるたびに、報酬、ε、直近の損失を表示します。

```text
episode=3 step=2451 reward=672.0 epsilon=0.956 loss=0.3821
```

これで、学習が止まっていないか、εが下がっているかを確認できます。ただし、1回の報酬や損失だけで良し悪しを判断する必要はありません。強化学習の数字は、わりと気分屋です。

## 8. 完成したコード

完成したコードは[07-training/main.py](./07-training/main.py)です。第6章までに作ったクラスや関数も含め、次の処理が1本につながっています。

1. 使用するデバイスを選ぶ
2. Q-NetworkとTarget Networkを作る
3. ε-greedyで行動を選ぶ
4. Replay Memoryへ経験を保存する
5. 4ステップごとにQ-Networkを学習させる
6. 1000ステップごとにTarget Networkを同期する
7. 10,000ステップごとにモデルを保存する

プロジェクトのルートから実行します。

```bash
uv run python 07-training/main.py
```

学習中はゲーム画面を表示しません。描画にも時間がかかるため、学習を優先するためです。進み具合はターミナルへ表示されます。

```text
使用デバイス: mps
episode=1 step=812 reward=324.0 epsilon=0.985 loss=--
episode=2 step=1617 reward=541.0 epsilon=0.971 loss=0.4382
...
モデルを保存しました: 07-training/q_network.pt
```

環境やランダムな行動によって、表示される数字は変わります。学習を止めるときは`Ctrl + C`を押してください。終了処理で、その時点のモデルが保存されます。

:::note info
最初は`TOTAL_STEPS = 2000`程度に減らし、エラーなく最後まで動くことを確認するのがおすすめです。動作を確認できたら、`100_000`へ戻して長時間学習させましょう。
:::

## 9. 保存したモデルで推論する

学習が終わったら、保存した`q_network.pt`を読み込み、マリオを操作させてみましょう。

学習中と推論中では、行動の選び方が少し違います。

| 学習中 | 推論中 |
| --- | --- |
| ε-greedyでランダム行動も試す | Q値が一番高い行動を選ぶ |
| Q-Networkを更新する | Q-Networkを更新しない |
| 画面を表示しない | 画面を表示する |

推論では探索する必要がないため、εは使いません。学習したQ-Networkが「これが一番よい」と判断した行動だけを選びます。

まず、学習コードの`make_env()`へ`render_mode`を渡せるように変更します。

```python
def make_env(render_mode="rgb_array"):
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        render_mode=render_mode,
    )
    # 以下は同じ
```

学習時はデフォルトの`"rgb_array"`、推論時はウィンドウを表示する`"human"`を使えます。

次に、`07-training`フォルダへ`inference.py`を作ります。

```python:inference.py
import time

import torch

from main import MODEL_PATH, QNetwork, get_device, make_env, states_to_tensor


def load_q_network(env, device, model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {model_path}\n"
            "先に main.py を実行してモデルを作成してください。"
        )

    q_network = QNetwork(n_actions=env.action_space.n).to(device)
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    q_network.load_state_dict(state_dict)
    q_network.eval()
    return q_network


def play(env, q_network, device, max_steps=10_000):
    state, info = env.reset()

    try:
        for _ in range(max_steps):
            state_tensor = states_to_tensor([state], device)

            with torch.no_grad():
                q_values = q_network(state_tensor)

            action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(4 / 60)

            if terminated or truncated:
                state, info = env.reset()
            else:
                state = next_state
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    device = get_device()
    env = make_env(render_mode="human")
    q_network = load_q_network(env, device)

    print(f"使用デバイス: {device}")
    print(f"読み込んだモデル: {MODEL_PATH}")

    play(env, q_network, device)


if __name__ == "__main__":
    main()
```

### 保存した値をQ-Networkへ戻す

```python
state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)
q_network.load_state_dict(state_dict)
```

`torch.load()`でファイルを読み込み、`load_state_dict()`で新しく作ったQ-Networkへ学習済みの値を入れます。

`map_location=device`を指定すると、CPUで学習したモデルをGPUで動かす場合や、その逆の場合にも、現在使うデバイスへ読み込めます。

### 学習済みの行動だけを選ぶ

推論ループでは、ε-greedyの代わりに、Q値が一番高い行動を毎回選びます。

```python
with torch.no_grad():
    q_values = q_network(state_tensor)

action = q_values.argmax(dim=1).item()
```

この部分は、第4章で作った未学習モデルの推論と同じです。推論方法は変わらず、Q-Networkの中身だけが学習済みの値へ変わっています。

プロジェクトのルートから実行します。

```bash
uv run python 07-training/inference.py
```

ウィンドウが開き、保存したQ-Networkがマリオを操作します。終了するときは`Ctrl + C`を押してください。

### SSH先で推論する

Ubuntu ServerへSSHで接続している場合、`render_mode="human"`を使っても手元の画面にゲームウィンドウは表示されません。サーバーは遠くで元気にマリオを動かしていますが、こちらからは見えない。少し寂しい状態です。

そこで、ウィンドウを開かずに推論する[inference_cli.py](./07-training/inference_cli.py)も用意します。

```bash
uv run python 07-training/inference_cli.py
```

CLI版では`render_mode="rgb_array"`を使います。ゲーム画面はQ-Networkへ入力しますが、ウィンドウには表示しません。その代わり、途中経過をターミナルへ表示します。

```text
使用デバイス: cuda
読み込んだモデル: 07-training/q_network.pt
画面は表示せず、ターミナルへ途中経過を表示します
step=25 episode=1 x=223 time=395 score=0 action=4:right+A+B q=139.37
episode=1 result=終了 reward=239.0 max_x=304 time=393 score=0
actions: 4:right+A+B=23回, 1:right=7回, 3:right+B=5回
```

途中経過では、現在位置、残り時間、スコア、選んだ行動、その行動のQ値を確認できます。エピソードが終わると、最も遠くまで進んだ位置と、各行動を何回選んだかも表示します。その場で同じ行動を繰り返していないか調べるときにも便利です。

実行するステップ数と表示間隔は、オプションで変更できます。

```bash
uv run python 07-training/inference_cli.py \
    --max-steps 1000 \
    --log-interval 10
```

通常は待ち時間なしで推論するため、ゲームはできるだけ速く進みます。ゆっくり経過を眺めたい場合は、`--delay`で1ステップごとの待ち時間を指定できます。SkipFrameで1ステップにつき最大4フレーム進むため、実時間に近づけるなら約`0.067`秒です。

```bash
uv run python 07-training/inference_cli.py --delay 0.067
```

終了するときは、ウィンドウ版と同じく`Ctrl + C`を押します。

短い動作確認だけで保存したモデルでは、学習前とほとんど変わらない可能性があります。まずはコードが動くことを確認し、その後で学習ステップを増やして変化を比べてみましょう。急に世界記録を出さなくても大丈夫です。最初の目標は、最初のクリボーより少し長く生きることです。

## 第7章のまとめ

この章では、DQNの部品を学習ループへまとめました。

1. εを少しずつ下げながら、探索と活用を切り替えた
2. 最初の1000ステップは経験集めに専念した
3. Replay Memoryから経験を取り出し、Q-Networkを繰り返し更新した
4. Target Networkを定期的に最新の状態へ同期した
5. 学習したQ-Networkをファイルへ保存した
6. 保存したQ-Networkを読み込み、ウィンドウまたはCLIでマリオを操作した

これで、マリオの学習から推論までが1本につながりました。学習時間や設定を変えながら、学習前と学習後で動きがどう変わるか観察してみましょう。

<!-- 第7章ドラフトここまで -->
