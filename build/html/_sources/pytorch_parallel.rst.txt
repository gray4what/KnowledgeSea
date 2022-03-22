##########################
Pytorch parallel computing
##########################


**************
Apex
**************

**Apex = mixed precision + torch.distributed**

* initial 

.. code-block:: python

    from apex import amp

    model, optimizer = amp.initialize(model, optimizer)


* compare with torch.distributed:

.. code-block:: python

    .parallel import DistributedDataParallel

    ## Apex
    model = DistributedDataParallel(model)  # save args
    # # torch.distributed
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


* warp loss:

.. code-block:: python

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()


*************
sample code
*************

.. code-block:: python

    # main.py
    import torch
    import argparse
    import torch.distributed as dist

    from apex.parallel import DistributedDataParallel

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    train_dataset = ...
    #
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

    model = ...

    ####
    model, optimizer = amp.initialize(model, optimizer)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = optim.SGD(model.parameters())

    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ...
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

    # run torch.distributed.launch to start
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
