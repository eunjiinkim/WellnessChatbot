from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

model = GPT2LMHeadModel.from_pretrained('./finetuned_model')

U_TKN = '<usr>'
S_TKN = '<sys>'
EOS = '</s>'
MASK = '<unused0>'
PAD = '<pad>'
SENT = '<unused1>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

parser = argparse.ArgumentParser(description='MentalHealth-bot based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

args = parser.parse_args()


def chat():
    with torch.no_grad():
        qs=[]
        while 1:
            q = input('나 > ').strip()
            qs.append(q) # history 저장

            if q == 'quit':
                break
            a=''
            user = U_TKN + q + SENT + a
            encoded = tokenizer.encode(user)
            input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
            output = model.generate(input_ids,max_length=50,
                                         num_beams=10, do_sample=False, 
                                         top_k=50, no_repeat_ngram_size=2,
                                        temperature=0.85)
            a=tokenizer.decode(output[0])
            idx = torch.where(output[0]==tokenizer.encode('<sys>')[0])
            chatbot = tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True)
 
            if '답변' in a: # 응, 아니 등이 input으로 들어왔을 때
                a_new = ''
                user = U_TKN + ''.join(qs[-2:]) + SENT + a_new # 직전 history 가지고 와서 sentiment 고려해주기
                encoded = tokenizer.encode(user)
                input_ids = torch.LongTensor(encoded).unsqueeze(dim=0)
                output = model.generate(input_ids,max_length=50,
                                         num_beams=10, do_sample=False, 
                                         top_k=50, no_repeat_ngram_size=2,
                                        temperature=0.85)
                a_new = tokenizer.decode(output[0], skip_special_tokens=True)
                idx = torch.where(output[0]==tokenizer.encode('<sys>')[0])
                chatbot = tokenizer.decode(output[0][int(idx[0])+1:], skip_special_tokens=True)
                
                print("챗봇 > {}".format(chatbot.strip()))

            else: 
                print("챗봇 > {}".format(chatbot.strip()))

                
if __name__ == "__main__":
    if args.chat:
        chat()
