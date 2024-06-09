Obtain Magnetic Laplacian PE via PyG Pretransform
==================================================

We provide a function that could obtain magnetic Laplacian PE based on `torch_geometric.transforms <https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html>`_, our codes is built on `AddLaplacianEigenvectorPE <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html#torch_geometric.transforms.AddLaplacianEigenvectorPE>`.

The class is located at ``_ and is as follows:

.. code_block:: python

    @functional_transform('add_mag_laplacian_eigenvector_pe')
    class AddMagLaplacianEigenvectorPE(BaseTransform):
        r"""Adds the Magnetic Laplacian eigenvector positional encoding. The eigenvectors are
        complex number, so choosing k of them means there will be 2*k channels (k real parts and k imaginary parts)
        in total.
    
        Args:
            k (int): The number of non-trivial eigenvectors to consider.
            attr_name (str, optional): The attribute name of the data object to add
                positional encodings to. If set to :obj:`None`, will be
                concatenated to :obj:`data.x`.
                (default: :obj:`"laplacian_eigenvector_pe"`)
            **kwargs (optional): Additional arguments of
                :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
                :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
                :attr:`is_undirected` is :obj:`True`).
        """
        def __init__(
                self,
                k: int,
                q: float = 0.1,
                dynamic_q: bool = False,
                multiple_q: int = 1,
                attr_name: Optional[str] = 'laplacian_eigenvector_pe',
                **kwargs,
        ):
            self.k = k
            self.q = q
            self.dynamic_q = dynamic_q
            self.multiple_q = multiple_q
            self.attr_name = attr_name
            self.kwargs = kwargs
    
        def __call__(self, data: Data) -> Data:
            from scipy.sparse.linalg import eigs, eigsh
            eig_fn = eigsh # always use hermitian version
    
            num_nodes = data.num_nodes
            edge_index, edge_weight_list = get_mag_laplacian(
                data.edge_index,
                data.edge_weight,
                normalization='sym',
                num_nodes=num_nodes,
                q = self.q,
                dynamic_q=self.dynamic_q,
                multiple_q=self.multiple_q
            )
    
            pe_list = []
            eigvals_list = []
            for edge_weight in edge_weight_list:
                L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    
                #try:
                #    eig_vals, eig_vecs = eig_fn(
                #        L,
                #        k=self.k,
                #        which='SA',
                #        return_eigenvectors=True,
                #        **self.kwargs,
                #    )
                #    sort = eig_vals.argsort()
                #    eig_vals = eig_vals[sort]
                #    eig_vecs = eig_vecs[:, sort]
                #except:
                    #from scipy.linalg import eigh
                    #eig_vals, eig_vecs = eigh(L.toarray())
                    #sort = eig_vals.argsort()[:self.k]
                    #eig_vals = eig_vals[sort]
                    #eig_vecs = eig_vecs[:, sort]
                    #eig_vals = eig_vals[:self.k]
                    #eig_vecs = eig_vecs[:, :self.k]
    
                #if np.isnan(eig_vecs).any() or np.isnan(eig_vals).any():
                eig_vals, eig_vecs = np.linalg.eigh(L.toarray())
                sort = eig_vals.argsort()[:self.k]
                eig_vals = eig_vals[sort]
                eig_vecs = eig_vecs[:, sort]
    
                # padding zeros if num of nodes less than desired pe dimension
                if len(eig_vals) < self.k:
                    eig_vals = np.pad(eig_vals, (0, self.k - len(eig_vals)))
                    eig_vecs = np.pad(eig_vecs, ((0, 0),(0, self.k - eig_vecs.shape[-1])))
    
                #pe = np.concatenate( (np.expand_dims(np.real(eig_vecs[:, eig_vals.argsort()]), -1),
                #                           np.expand_dims(np.imag(eig_vecs[:, eig_vals.argsort()]), -1)), axis=-1)
                #pe = np.concatenate( (np.expand_dims(np.real(eig_vecs), -1),
                #                           np.expand_dims(np.imag(eig_vecs), -1)), axis=-1)
                # pe = torch.from_numpy(pe) # [N, pe_dim, 2]
                #sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
                #sign = torch.unsqueeze(torch.unsqueeze(sign, dim=-1), dim=0)
                #pe = sign * pe
    
                #pe = pe.flatten(1, 2) # [N, pe_dim * 2]
    
    
    
                pe = torch.from_numpy(np.expand_dims(eig_vecs, 1))
                eig_vals = np.expand_dims(np.expand_dims(eig_vals, 0), 0)
                pe_list.append(pe)
                eigvals_list.append(torch.from_numpy(eig_vals))
            #pe = torch.cat(pe_list, dim=-1)
            #eig_vals = torch.cat(eigvals_list, dim=-1)
            pe = torch.cat(pe_list, dim=1)
            eig_vals = torch.cat(eigvals_list, dim=1)
            data = add_node_attr(data, pe, attr_name=self.attr_name)
            #data = add_node_attr(data, eig_vals.reshape(1, -1), attr_name='Lambda')
            data = add_node_attr(data, eig_vals, attr_name='Lambda')
            return data

