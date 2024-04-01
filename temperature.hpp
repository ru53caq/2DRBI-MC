// Parallel Tempering for ALPSCore simulations
// Copyright (C) 2019  Jonas Greitemann

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <string>

#include <alps/params.hpp>

namespace phase_space_point {
    struct temperature {
        static const size_t label_dim = 1;
        using iterator = double *;
        using const_iterator = double const *;

        static void define_parameters(alps::params & params, std::string prefix="") {
            params.define<double>(prefix + "temperature", 1., "temperature");
        }

        static bool supplied(alps::params const& params, std::string prefix="") {
            return params.supplied(prefix + "temperature");
        }

        temperature() : temp(-1) {}
        temperature(double temp) : temp(temp) {}
        temperature(alps::params const& params, std::string prefix="")
            : temp(params[prefix + "temperature"].as<double>()) {}

        template <class Iterator>
        temperature(Iterator begin) : temp(*begin) {}

        const_iterator begin() const { return &temp; }
        iterator begin() { return &temp; }
        const_iterator end() const { return &temp + 1; }
        iterator end() { return &temp + 1; }

        double temp;
    };

    template <typename Point>
    bool operator== (Point const& lhs, Point const& rhs) {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <typename Point>
    bool operator!= (Point const& lhs, Point const& rhs) {
        return !(lhs == rhs);
    }

    template <typename Point>
    bool operator< (Point const& lhs, Point const& rhs) {
        return std::lexicographical_compare(
            lhs.begin(), lhs.end(),
            rhs.begin(), rhs.end());
    }

}