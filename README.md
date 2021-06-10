# WellnessChatbot for Mental Health
정신건강을 위한 한국어 상담 챗봇입니다 🙂
(2021.06.10) 현재 실행 가능한 [Demo](https://share.streamlit.io/eunjiinkim/wellnesschatbot/main)입니다.

fine-tuning한 모델을 [huggingface🤗](https://huggingface.co/eunjin/kogpt2-finetuned-wellness)에 업로드하였습니다.

# Based on KoGPT2 (v2)
KoGPT2 기반 일상대화 챗봇(https://github.com/haven-jeon/KoGPT2-chatbot) 의 코드를 참조하여 재구성하였습니다.

# Datasets
### Chit-chat datasets
더욱 자연스러운 대화를 위해 위 코드에서 사용한 일상 데이터셋을 함께 구성하였습니다.
* 제시된 sentiment label을 따라 '일상', '사랑', '이별'로 주었습니다.

|user|system| sentiment|
|-------|-------|-----|
|12시 땡!|하루가 또 가네요. |일상|
|1지망 학교 떨어졌어|위로해 드립니다.|일상|
|3박4일 놀러가고 싶다|여행은 언제나 좋죠.|일상|

### Wellness datasets
AI hub에서 제공하는 wellness 정신건강 상담 챗봇 데이터셋을 가공하였습니다.
* wellness 1: 359개의 sentiment label
* wellness 2: 19개의 sentiment label
    * 대화의 turn을 포함하고 있는 것이 특징입니다.
    * 우선 user-system 쌍을 1:1로 이루어야하기 때문에 각 발화들을 turn에 맞추어 재구성하였습니다.
* 1&2에서 모두 sentiment label을 자연언어 그대로 가져왔으며, turn에 따른 사용자 발화 중 '응', '아니'와 같은 답변을 <긍정답변> <부정답변>의 label로 전환하였습니다.

|user|system| sentiment|
|-------|-------|-----|
|너무 불안한 거 있지.|불안감을 느끼고 계시는군요. 00님이 불안감으로 많이 힘드신 건 아닌지 걱정이 돼요.|불안|
|완전 힘들고 짜증나요.|	평소 즐기던 일이 더 이상 즐겁지 않거나 흥미가 사라졌나요?|무기력|
|다 때려치고 싶어.|마음이 잠깐 지쳤나 봐요. 아무 의미 없는 일은 없을 거예요.|의욕상실|
|친구들이 자꾸 차갑게 말 하니까 대화하기 싫어서 피하게 되더라|매사에 주눅들고 자신감이 없으신가요?|자존감저하|
|어 맞아|그러셨군요. 많이 당황스러우셨을 것 같아요. 이런 상황이 오래 지속될까 걱정돼요.|긍정답변|
|그런 것 같아요|그렇군요. 많이 지치신 것 같아 걱정이에요. 질문을 하나 드릴게요.|긍정답변|
|그건 아닌데|그렇군요. 불안을 줄이는 방법으로 꾸준한 운동이나 명상, 규칙적인 수면이 도움이 될 수 있어요. (...) |부정답변|
|그렇진 않은데|그렇군요. 불안을 줄이는 방법으로 꾸준한 운동이나 명상, 규칙적인 수면이 도움이 될 수 있어요.|부정답변|

# Architecture

![그림1](https://user-images.githubusercontent.com/55074165/121313370-486b9780-c941-11eb-9b2f-b2dac1d6edae.png)

* `<usr>사용자 발화` `sentiment 정보` `챗봇 답변` 이 주어졌을 때 모델이 sentiment 정보를 생성한 후 그에 맞는 답변을 생성할 수 있도록 학습하였습니다.
* sentiment가 <긍정답변> 혹은 <부정답변>을 생성하면 이전 발화 history과 합쳐 다시 input을 생성합니다.
* 따라서 context를 유지한채 대화의 turn을 이어갈 수 있습니다.

# Chat Example

```bash
나 > 고민이 많다
챗봇 > 마음이 예민해져서 매우 괴로우셨겠어요. 저랑 차분하게 잠시 대화를 해봐요.
나 > 공부하는게 힘들어
챗봇 > 공부하면 더 많은 선택을 할 수 있죠.
나 > 그치만 너무 힘든걸
챗봇 > 조금만 더 버텨보세요.
나 > 싫어 너무 무기력해
챗봇 > 평소 즐기던 일이 더 이상 즐겁지 않거나 흥미가 사라졌나요?
나 > 응 아무것도 하기 싫어
챗봇 > 아무것도 하고 싶지 않고, 하루가 길게 느껴지나요.
나 > 맞아
챗봇 > 00님이 힘든 것 같아 걱정이에요. 더 얘기해볼까요?
나 > 성적이 자꾸 떨어져
챗봇 > 노력한 만큼 결과가 나오지 않으면 속상하죠.
나 > 친구들이랑도 잘 안맞는것같아
챗봇 > 사람들이 나에게 차갑게 대하는 것 같으신가요?
나 > 응응
챗봇 > 자신을 더 사랑해주세요.
나 > 친구들이 나를 안좋아하는것같아
챗봇 > 자존감이 떨어져 속상하고 고민이 많이 드셨겠어요.
```

generate setting에 따라 결과가 달라집니다.

# How to train

```bash
python3 trainer.py --train --gpus 3 --max_epochs 3
```
multi-gpu를 사용하였으며, `--gpus 1` 설정시 오류가 난다면 trainer.py에서 accelerator="dp"를 지워보시기 바랍니다.

# How to chat

```bash
python3 chat.py --chat
```
fine-tuning한 모델이 directory 안에 저장되어있어야 합니다.

# Reference
* KoGPT2 chitchat: https://github.com/haven-jeon/KoGPT2-chatbot
* wellness datasets: https://aihub.or.kr/keti_data_board/language_intelligence
* Chitchat datasets: https://github.com/songys/Chatbot_data
