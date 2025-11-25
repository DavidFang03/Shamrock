// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AddForceAntiPressurer2.hpp
 * @author David Fang (david.fang@ikmail.com) --no git blame--
 * @brief Adds the acceleration for a fictitious force to counter pressure
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::common::modules {

    template<class Tvec>
    class AddForceAntiPressurer2 : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        AddForceAntiPressurer2() = default;

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shamrock::solvergraph::IDataEdge<Tscal> &rho_0;
            const shamrock::solvergraph::IDataEdge<Tscal> &rmin;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_positions;
            const shamrock::solvergraph::Indexes<u32> &sizes;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_accel_ext;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> rho_0,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> rmin,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_positions,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_accel_ext) {
            __internal_set_ro_edges({constant_G, rho_0, rmin, spans_positions, sizes});
            __internal_set_rw_edges({spans_accel_ext});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(4),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        void _impl_evaluate_internal() override {
            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.spans_positions.check_sizes(edges.sizes.indexes);
            edges.spans_accel_ext.ensure_sizes(edges.sizes.indexes);

            Tscal rho0 = edges.rho_0.data;
            Tscal G    = edges.constant_G.data;
            Tscal rmin = edges.rmin.data;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.spans_positions.get_spans()},
                sham::DDMultiRef{edges.spans_accel_ext.get_spans()},
                edges.sizes.indexes,
                [mrho0G = -rho0 * G, rmin](u32 gid, const Tvec *xyz, Tvec *axyz_ext) {
                    Tvec r_a     = xyz[gid] - rmin;
                    Tscal abs_ra = sycl::length(r_a);
                    Tscal acc    = mrho0G * (1 / rmin - 1 / abs_ra);
                    axyz_ext[gid] += {acc / 3, acc / 3, acc / 3};
                });
        };
        inline virtual std::string _impl_get_label() const override {
            return "AddForceAntiPressurer2";
        };
        inline virtual std::string _impl_get_tex() const override { return "TODO"; };
    };

} // namespace shammodels::common::modules
