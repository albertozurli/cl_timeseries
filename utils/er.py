

    def observe(self, inputs, labels,not_aug_inputs):
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param not_aug_inputs: some methods could require additional parameters
        :return: the value of the loss function
        """
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs,buf_labels = self.buffer.get_data(self.args.minibatch_size)
            inputs = torch.cat((inputs,buf_inputs))
            labels = torch.cat((labels,buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs,labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,labels= labels[:real_batch_size])

        return loss.item()
