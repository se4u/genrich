import sys, numpy.linalg, logging, numpy, signal, time, random
from util_oneliner import batcher, ensure_dir, tictoc, mean, get_vocab_from_file

global starting_time
global sentence_done
global options
global mo_model
starting_time=None
sentence_done=None
options=dict(e.split('=') for e in sys.argv[1:])
random.seed(int(options['RNG_SEED']))
mo_model=None
log_filename=options['SAVE_FILE']+".logger"
ensure_dir(log_filename)
sys.stderr.write('Logging output to file: %s\n'%log_filename)
logging.basicConfig(filename=log_filename,level=logging.DEBUG)
logger=logging

def signal_handler(signal, frame):
    """ Catch Ctrl+C and exit after printing the time taken and the
    sentences processed 
    """
    logger.debug('\nTime: %0.1f, Sentence: %d\n'%(
            time.time()-starting_time, sentence_done))
    # Save the model at whatever state we are in
    # that's the beauty of sgd
    ensure_dir(options['SAVE_FILE'])
    mo_model.save(options['SAVE_FILE'])
    logger.debug("Saved model to %s"%options['SAVE_FILE'])
    sys.exit(1)
    return
signal.signal(signal.SIGINT, signal_handler)

def tune_batch_size_learning_rate(mo_model,
                                  sup_train_fn,
                                  validation_fn,
                                  batch_sizes=(10, 5, 3, 1),
                                  learning_rates=tuple(pow(5,-i)
                                                       for i in range(6,0,-1)),
                                  quota=500):
    """ Tune the batch size and learning rate over the range that is supplied.
    """
    d={}
    initial_values=[e.get_value() for e in mo_model.params]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            with tictoc('Tuning with Batch size : %d, Learning rate: %0.2e'%(batch_size, learning_rate)):
                for batch_idx, batch in enumerate(batcher(open(sup_train_fn, 'rb'), batch_size)):
                    if batch_idx*batch_size > quota:
                        break

                    words=[[e.split('/')[0] for e in row]
                           for row in batch]
                    tags=[[e.split('/')[1] for e in row]
                          for row in batch]

                    iter_update_gradient=mo_model.batch_update_ao(
                        learning_rate, tags, words)
                    logger.debug(' '.join([str(e) for e in [batch_size, learning_rate, batch_idx]]) +
                                 ' '.join([str(numpy.linalg.norm(e)) for e in iter_update_gradient]))
            # Now that we have used up all the batches let's see where we reach on the validation set
            correct_tags=0.0
            total_tags=0.0
            with open(validation_fn, "rb") as vf:
                for row in vf:
                    row=row.strip().split()
                    words=[e.split("/")[0] for e in row]
                    true_tags=[e.split("/")[1] for e in row]
                    [predicted_tags, predicted_scores]= \
                        mo_model.predict_posterior_tag_sequence(words)
                    assert(len(predicted_tags)==len(true_tags))
                    correct_tags+=sum(1 if a==b else 0 for a,b in
                                     zip(true_tags,predicted_tags))
                    total_tags+=len(true_tags)
            d[batch_size,learning_rate]=float(correct_tags)/total_tags
            logging.debug('The validation done so far %s'%str(d))
            # Now reset to initial values to do a fair comparison between different settings
            # Don't borrow anything here. I dont want any memory corruption at all.
            for i in xrange(len(initial_values)):
                mo_model.params[i].set_value(initial_values[i])
    # print d
    logger.debug('The validation accuracy of different batches and learning rates are the following: '+str(d))
    ((batch_size, learning_rate),ve)=max(d.items(), key=lambda x: x[1])
    logger.debug('The optimal batch_size: %d, learning_rate: %f, validation error: %f'%(
            batch_size, learning_rate, ve))
    return [batch_size, learning_rate]

# fn means filename
def train_model(mo_model,
                sup_train_fn,
                unsup_train_fn,
                validation_fn,
                batch_size,
                epoch,
                learning_rate,
                optimization_method,
                validation_freq,
                validation_action):
    """ Train the model using the following logic
    batch_size = Size of the minibatch used to caclulate the gradient.
    epoch = the number of time to go over the data
    learning_rate = the initial learning rate
    optimization_method = sgd, adagrad etc. It decides what's
                          done to the gradients for the update.
                          It can also tell us how to
                          update the learning rates used. 
    validation_freq = how often we validate
    validation_action =  what we do after validation

    This method can be made a lot more ornate. But for starters I am
    implementing a simple grid search over a small portion of the
    training and validation data to automatically set the learning
    rate and the 
    """
    global sentence_done
    if batch_size==0 and learning_rate==0:
        """ Automatically figure out the optimum batch size and
            learning rate by doing a grid search for them """
        [batch_size, learning_rate]=tune_batch_size_learning_rate(
            mo_model,sup_train_fn,validation_fn)
    for sentence_done, row in enumerate(open(sup_train_fn, 'rb')):
        sys.stderr.write('.')
        row=row.strip().split()
        if len(row)==0:
            continue
        words=[e.split('/')[0] for e in row]
        tags=[e.split('/')[1] for e in row]
        with tictoc('Training update time'):
            iter_update_gradient=mo_model.update_ao(eta=learning_rate, tags=tags, words=words)
        
        if sentence_done%validation_freq==0:
            logger.debug(' '.join([str(numpy.linalg.norm(e)) for e in iter_update_gradient]))
            with tictoc('Prediction time'):
                prediction=mo_model.predict_posterior_tag_sequence(words)
                predicted_tags=prediction[0]
                predicted_scores=prediction[1]
                logger.debug("Mean Accuracy %f",
                              mean(map(lambda x,y: 1 if x==y else 0,
                                   predicted_tags, tags)))
                logger.debug('\n'.join(['/'.join([a,b,c,str(d)])
                                 for a,b,c,d
                                 in zip(words, tags,
                                        predicted_tags,predicted_scores)]))
        
        if any(numpy.isnan(e).any() for e in iter_update_gradient):
            raise Exception('iter_update_gradient became nan!')
    return mo_model

if __name__=='__main__':
    model_type = options['MODEL_TYPE']
    if model_type=='order0hmm':
        from model.tag_order0hmm import tag_order0hmm
        cl_model = tag_order0hmm
    elif model_type=='rhmm':
        raise NotImplementedError
        from model.tag_rhmm import tag_rhmm
        cl_model = tag_rhmm
    else:
        raise NotImplementedError
    with tictoc('Init mo_model'):
        mo_model=cl_model(cpd_type=options['CPD_TYPE'],
                      objective_type=options['OBJECTIVE_TYPE'],
                      param_reg_type=options['PARAM_REG_TYPE'],
                      param_reg_weight=float(options['PARAM_REG_WEIGHT']),
                      unsup_ll_weight=float(options['UNSUP_LL_WEIGHT']),
                      init_from_file=int(options['INIT_FROM_FILE']),
                      param_filename=options['PARAM_FILENAME'],
                      tag_vocab=\
                          get_vocab_from_file(open(options['TAG_VOCAB_FILE'],
                                                   'rb')),
                      word_vocab=\
                          get_vocab_from_file(open(options['WORD_VOCAB_FILE'],
                                                   'rb')),
                      test_time=False)
    with tictoc('Training mo_model'):
        train_model(mo_model=mo_model,
                    sup_train_fn=options['SUP_TRAIN_FILE'],
                    unsup_train_fn=options['UNSUP_TRAIN_FILE'],
                    validation_fn=options['VALIDATION_FILE'],
                    batch_size=int(options['BATCH_SIZE']),
                    epoch=int(options['EPOCH']),
                    learning_rate=float(options['LEARNING_RATE']),
                    optimization_method=options['OPTIMIZATION_METHOD'],
                    validation_freq=int(options['VALIDATION_FREQ']),
                    validation_action=options['VALIDATION_ACTION'])
    
    with tictoc('Saving model'):
        ensure_dir(options['SAVE_FILE'])
        mo_model.save(options['SAVE_FILE'])

