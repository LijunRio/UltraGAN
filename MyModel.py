def define_nets(self):
    self.encoder = self.ENCODERS[self.cfg["ENCODER"]](vocab_size=self.t2i_dataset.vocab_size,
                                                      embed_size=self.cfg["E_EMBED_SIZE"],
                                                      hidden_size=self.cfg["E_HIDEN_SIZE"],
                                                      max_len=[self.t2i_dataset.max_len_finding,
                                                               self.t2i_dataset.max_len_impression],
                                                      unit=self.cfg["RNN_CELL"],
                                                      feature_base_dim=self.cfg["D_CHANNEL_SIZE"]
                                                      ).to(self.device)

    decoders_F = []
    first_decoder = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                       feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                       uprate=self.base_ratio).to(self.device)
    # first_decoder.apply(init_weights)
    decoders_F.append(first_decoder)
    for i in range(1, self.P_ratio + 1):
        nf = 128
        pdecoder = self.P_DECODER[self.cfg["PDECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                        feature_base_dim=nf).to(self.device)
        # pdecoder.apply(init_weights)
        decoders_F.append(pdecoder)

    self.decoder_F = MultiscaleDecoder(decoders_F)

    decoders_L = []
    first_decoder = self.DECODERS[self.cfg["DECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                       feature_base_dim=self.cfg["D_CHANNEL_SIZE"],
                                                       uprate=self.base_ratio).to(self.device)
    # first_decoder.apply(init_weights)
    decoders_L.append(first_decoder)
    for i in range(1, self.P_ratio + 1):
        nf = 128
        pdecoder = self.P_DECODER[self.cfg["PDECODER"]](input_dim=self.cfg["D_CHANNEL_SIZE"],
                                                        feature_base_dim=nf).to(self.device)
        # pdecoder.apply(init_weights)
        decoders_L.append(pdecoder)

    self.decoder_L = MultiscaleDecoder(decoders_L).to(self.device)

    self.embednet = Classifinet(backbone='resnet18').to(self.device)