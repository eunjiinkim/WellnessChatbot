#!/usr/bin/env python
# coding: utf-8


import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

model = GPT2LMHeadModel.from_pretrained('eunjin/kogpt2-finetuned-wellness')

U_TKN = '<usr>'
S_TKN = '<sys>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
SENT = '<unused1>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 



@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def chat(utterance,maxlen,numbeams,sampling,topk,ngramsize,temp):
    with torch.no_grad():
            qs=[]
            q = utterance
            qs.append(q) # history 저장

            a=''
            user = U_TKN + q + SENT + a
            encoded = tokenizer.encode(user)
            input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
            output = model.generate(input_ids,max_length=maxlen,
                                         num_beams=numbeams, do_sample=sampling, 
                                         top_k=topk, no_repeat_ngram_size=ngramsize,
                                        temperature=temp)
            a=tokenizer.decode(output[0])
            idx = torch.where(output[0]==tokenizer.encode('<sys>')[0])
            chatbot = tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True)
 
            if '답변' in a: # 응, 아니 등이 input으로 들어왔을 때
                a_new = ''
                user = U_TKN + ''.join(qs[-2:]) + SENT + a_new # 직전 history 가지고 와서 sentiment 고려해주기
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(input_ids,max_length=maxlen,
                                         num_beams=numbeams, do_sample=sampling, 
                                         top_k=topk, no_repeat_ngram_size=ngramsize,
                                        temperature=temp)
                a_new = tokenizer.decode(output[0], skip_special_tokens=True)
                idx = torch.where(output[0]==tokenizer.encode('<sys>')[0])
                chatbot = tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True)
                
                result = chatbot.strip()
            
            else: 
                result = chatbot.strip()
                
            

    return result


def main():

    # Title
    st.title("한국어 정신건강 상담 챗봇")
    st.subheader("")
    st.markdown("""
        🙂 wellness 챗봇 데이터(AI hub)와 일상대화 챗봇 데이터(github,@songys)을 kogpt2(skt)에 fine-tuning한 챗봇입니다.
        """
        """
        💜 각 챗봇 데이터의 감정sentiment 정보를 함께 생성하도록 학습하여 사용자에게 위로와 공감을 주도록 하였습니다.
        """)

    st.subheader("챗봇과 대화해보세요!")

    utterance = st.text_input("나: ", "오늘은 어떤 기분이신가요?")
    
    st.sidebar.subheader("Generation Settings")
    maxlen=st.sidebar.slider("max length of the sequence", 30, 60,value=50)
    numbeams=st.sidebar.slider("number of beams", 5,20, value=10)
    topk=st.sidebar.slider("top k sampling", 10, 50, value=20)
    ngramsize=st.sidebar.slider("Ngram size not to repeat", 1,4,value=2)
    temp=st.sidebar.slider("temperature: threshold probability of each token", 0.7, 0.95, value=0.85)
    sampling=st.sidebar.checkbox("do sampling", value=False)
    
    
    st.sidebar.subheader("About App")
    st.sidebar.text("Streamlit")
    
    if st.button("전송"):
        result = chat(utterance,maxlen,numbeams,sampling,topk,ngramsize,temp)
        st.text_area("챗봇: ", value=result)

    st.sidebar.subheader("개발자")
    st.sidebar.text("김은진, jyej3154@snu.ac.kr")
    st.sidebar.text("서울대학교 언어학과 석사과정")
    
    st.subheader("참조")
    st.markdown("kogpt2 기반 챗봇: https://github.com/haven-jeon/KoGPT2-chatbot")
    st.markdown("pytorch lightning: https://www.pytorchlightning.ai/")
    st.markdown("huggingface transformers: https://huggingface.co/")
    st.markdown("wellness dataset: https://aihub.or.kr/keti_data_board/language_intelligence")
    
if __name__ == '__main__':
	main()

