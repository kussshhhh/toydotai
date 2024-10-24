import torch 

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

print(len(text))

print(text[:1000]) # first 1000 chars

chars = sorted(list(set(text)))
print(''.join(chars))
vocab_size = len(chars)
print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s ]

decode = lambda l: ''.join(itos[i] for i in l) 

# print(encode("yo im kucchi"))
# print(decode(encode("yo im kucchi")))


data = torch.tensor(encode(text), dtype =torch.long )

print(data.shape)
print(data[:1000])
n = int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

block_size=8
print(train_data[:block_size+1])


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context=x[:t+1]
    target=y[t]
    print(f"when input is {context} the target is : {target}")