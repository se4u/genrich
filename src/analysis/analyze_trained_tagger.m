global filename;
filename = 'res/param_order4rhmm~lbl10~LL~L2~0.001~0.2~0~NONE~wsj_tag.train.tag.vocab~wsj_tag.train.word.vocabtrunc~wsj_tag.train_sup.head2000~wsj_tag.train_unsup~wsj_tag.minivalidate~1234~1~10~0.005~sgd~25000~NOACTION.mat';
load(filename);

get_pdf = @(pp) exp(pp)/sum(exp(pp));

get_tagp = @(TAG, pdf) pdf(strcmp(tag_vocab, TAG));
get_tagemb = @(TAG) tagemb(strcmp(tag_vocab, TAG),:);

get_wordp = @(WORD, pdf) pdf(strcmp(word_vocab, WORD));
get_wordemb = @(WORD) wordemb(strcmp(word_vocab, WORD),:);

get_k_random_samples = @(a,k) a(randint(1,k,length(a))+1);
%% First we find out the probability distribution over tags at the
%% start.
c = T1*S';
v = tagemb;
pdf = get_pdf(v*c);
[p, id]=max(pdf); disp(tag_vocab(id));
cellfun(@(TAG) get_p(TAG, pdf), ...
        {'NNP', 'DT', 'IN', '``', 'RB', 'NN', 'JJ', 'NNS', 'CC', ...
         'PRP', 'VBG', 'CD', ':', 'WRB', 'VBN'})
% ans =
%     0.2183    0.1701    0.1008    0.0349    0.0259
%     0.0670    0.0634    0.0290    0.0697    0.0243
%     0.0148    0.0025    0.0021    0.0033    0.0093

% By way of comparison. The MLE tag | Start pdf is the following
% make find_pdf_tag_given_S
% 0.2799 NNP , 0.2081 DT , 0.1374 IN , 0.0528 `` , 0.0502 RB ,
% 0.0473 NN , 0.0466 JJ , 0.0444 NNS , 0.0378 CC , 0.0180 PRP ,
% 0.0133 VBG , 0.0118 CD , 0.0056 : , 0.0055 WRB , 0.0049 VBN ,
% 0.0035 EX , 0.0032 PRP$ , 0.0032 WP , 0.0031 VB , 0.0030 TO ,
% 0.0028 ( , 0.0026 NNPS , 0.0019 RBR , 0.0017 JJS , 0.0017 LS ,
% 0.0016 VBZ , 0.0016 JJR , 0.0015 SYM , 0.0009 PDT , 0.0008 WDT ,
% 0.0008 VBP , 0.0006 VBD , 0.0006 UH , 0.0004 RBS , 0.0004 MD ,
% 0.0003 FW , 0.0002 $ , 0.0001 WP$

% But there is also a problem that the true sorted distribution has
% some weird quirks.
[sorted_pdf,tmp]=sort(pdf, 'descend'); tag_vocab(tmp)
% 'NNP', 'DT', 'IN', 'CC', 'NN', 'JJ', '``', 'NNS', 'RB', 'TO',
% 'PRP', 'PRP$', '''', '.', 'VBG', 'WDT', 'E', 'VBN', ',', 'POS',
% 'WP', '$', 'NNPS', 'JJR', 'WRB', 'VBZ', 'RBR', 'EX', 'CD', 'JJS',
% ':', 'RP', '(', 'VBD', 'PDT', ')', 'RBS', 'VB', 'UH', 'WP$',
% 'LS', '#', 'FW', 'MD', 'SYM', 'VBP'

% 0.2183, 0.1701, 0.1008, 0.0697, 0.0670, 0.0634, 0.0349, 0.0290,
% 0.0259, 0.0252, 0.0243, 0.0204, 0.0197, 0.0153, 0.0148, 0.0123,
% 0.0100, 0.0093, 0.0076, 0.0066, 0.0064, 0.0053, 0.0047, 0.0043,
% 0.0033, 0.0033, 0.0030, 0.0028, 0.0025, 0.0021, 0.0021, 0.0019,
% 0.0019, 0.0015, 0.0014, 0.0014, 0.0013, 0.0012, 0.0010, 0.0009,
% 0.0008, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003

% Basically it seems that probabilities of things that occur more
% "in general" are pumped up higher.

%% Now lets find the most probable words given "DT" and "s"
c=Tt1*get_tagemb('DT')'+Tt2*S';v=wordemb;pdf = get_pdf(v*c);
[sorted_pdf,tmp]=sort(pdf, 'descend'); word_vocab(tmp);

% MIPS based heuristic
IP = v*c;
Z_true = sum(exp(IP));
sIP = sort(IP, 'descend');
k=10;
top_sum = sum(exp(sIP(1:k)));
zero_sum = sum(exp(sIP(end-k:end)));
rest=sIP(11:end-11);
rest_sum = sum(exp(get_k_random_samples(rest, k)))/k*length(rest);
Z_hat = top_sum + zero_sum + rest_sum
relative_error = (Z_hat-Z_true)/Z_true
% ans = 
%     'the'
%     'a'
%     'some'
%     'this'

% sorted_pdf =
%     0.5970
%     0.3008
%     0.0290
%     0.0194
%% Sorted pdf over words given that the previous tags were S, NNS
c=Tt1*get_tagemb('NNS')'+Tt2*S';v=wordemb;pdf = get_pdf(v*c);
[sorted_pdf,tmp]=sort(pdf, 'descend'); word_vocab(tmp);

% MIPS based heuristic
IP = v*c;
Z_true = sum(exp(IP))
sIP = sort(IP, 'descend');
k=100;
top_sum = sum(exp(sIP(1:k)));
zero_sum = sum(exp(sIP(end-k:end)));
rest=sIP(k+1:end-k-1);
rest_sum = sum(exp(get_k_random_samples(rest, k)))/k*length(rest);
Z_hat = top_sum + zero_sum + rest_sum
relative_error = (Z_hat-Z_true)/Z_true
%% Now what I see is evidence that some learning has definitely
%% taken place. The pdfs are what they should be.
%% What I want is to create some hundredss of context embeddings
%% for tags and words and to see what the most probable word/tag
%% for those contexts is.